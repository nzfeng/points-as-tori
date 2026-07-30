# Text in these shaders can get replaced during execution based on values that change occasionally (and warrant the overhead of re-compiling the shader)

# Set up a pixel shader where a single quad takes up the whole screen.
# We shouldn't have to change this code.
VERTEX_SHADER = """
    #version 150 core
    
    in vec2 position;
    out vec2 fragCoord;
    out vec2 iResolution;

    uniform vec2 u_resolution;

    void main() {
        // Convert from normalized device coordinates to screen coordinates
        fragCoord = (position + 1.0) * 0.5 * u_resolution;
        iResolution = u_resolution;
        gl_Position = vec4(position, 0.0, 1.0);
    }
"""

COMMON_SHADER = """
const float PI = 3.14159265358979323846;
const float INFINITY = 100000000.0;

uniform sampler2D shape, isocontour; // texture that stores the ground-truth shape, if availablee
uniform sampler2D pointcloud; // texture that stores point cloud positions and normals
uniform sampler2D pointcolors;
uniform int neighborhood_size;
const int max_neighborhood_size = 128;
uniform float neighborhood_radius;
uniform int n_faces;
uniform int n_points;
uniform int texture_size; 
uniform float maxExpArg;
uniform float reg_epsilon;
uniform bool usePointAreas;

/*
 * erf is not a built-in function in GLSL, so approximate with elementary functions:
 * https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
 */
float erf(float x) {
    float x2 = x*x;
    float a = 0.147;
    return sign(x) * sqrt(1.0 - exp(-x2 * ((4.0/PI) + a*x2))/(1.0 + a*x2));
}

// ==== retrieve texture data

ivec2 index_to_texel(int idx) {
    int x = idx % texture_size;
    int y = idx / texture_size;
    return ivec2(x, y);
}

// Sample a texture.. 
// It is assumed that the i-th texel stores an attribute associated with the i-th entry of an array.
vec4 fetch_texel(int idx, sampler2D texture) {
    ivec2 texel_pos = index_to_texel(idx);
    return texelFetch(texture, texel_pos, 0);
}

vec3 fetch_point(int pIdx) {
    return fetch_texel(2*pIdx, pointcloud).xyz;
}

vec3 fetch_normal(int pIdx) {
    return fetch_texel(2*pIdx+1, pointcloud).xyz;
}

// ==== BVH

// point cloud
uniform sampler2D pc_bvh_nodes;
uniform sampler2D pc_bvh_prim_indices;
uniform int n_pc_bvh_nodes;

// mesh
uniform sampler2D mesh_bvh_nodes;
uniform sampler2D mesh_bvh_prim_indices;
uniform int n_mesh_bvh_nodes;

// isocontour
uniform sampler2D isomesh_bvh_nodes;
uniform sampler2D isomesh_bvh_prim_indices;
uniform int n_isomesh_bvh_nodes;

// Some of the BVH logic below was implemented with the help of Claude

void fetchBVHNode(sampler2D bvh_nodes, int nodeIdx, out vec3 aabb_min, out vec3 aabb_max,
                  out int left_idx, out int right_idx,
                  out int prim_start, out int prim_count) {
    int base_idx = nodeIdx * 3;
    
    vec4 data0 = fetch_texel(base_idx, bvh_nodes);
    vec4 data1 = fetch_texel(base_idx + 1, bvh_nodes);
    vec4 data2 = fetch_texel(base_idx + 2, bvh_nodes);
    
    aabb_min = data0.xyz;
    aabb_max = vec3(data0.w, data1.xy);
    left_idx = int(data1.z);
    right_idx = int(data1.w);
    prim_start = int(data2.x);
    prim_count = int(data2.y);
}

void fetchBVHNodeAABB(sampler2D bvh_nodes, int nodeIdx, 
                      out vec3 aabb_min, out vec3 aabb_max) {
    int base_idx = nodeIdx * 3;
    vec4 data0 = fetch_texel(base_idx, bvh_nodes);
    vec4 data1 = fetch_texel(base_idx + 1, bvh_nodes);
    aabb_min = data0.xyz;
    aabb_max = vec3(data0.w, data1.xy);
}

bool intersectAABB(vec3 ray_origin, vec3 ray_dir, vec3 aabb_min, vec3 aabb_max,
                   out float t_near, out float t_far) {
    vec3 inv_dir = 1.0 / (ray_dir + vec3(1e-10));
    vec3 t0 = (aabb_min - ray_origin) * inv_dir;
    vec3 t1 = (aabb_max - ray_origin) * inv_dir;
    
    vec3 t_min = min(t0, t1);
    vec3 t_max = max(t0, t1);
    
    t_near = max(max(t_min.x, t_min.y), t_min.z);
    t_far = min(min(t_max.x, t_max.y), t_max.z);
    
    return t_near <= t_far && t_far >= 0.0;
}

float pointAABBDistanceSq(vec3 point, vec3 aabb_min, vec3 aabb_max) {
    vec3 closest = clamp(point, aabb_min, aabb_max);
    vec3 diff = point - closest;
    return dot(diff, diff);
}

int bvhClosestPoint(vec3 query, out float min_dist) {
    min_dist = INFINITY;
    int closest_idx = -1;
    
    int stack[stack_size];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        
        vec3 aabb_min, aabb_max;
        int left_idx, right_idx, prim_start, prim_count;
        fetchBVHNode(pc_bvh_nodes, node_idx, aabb_min, aabb_max, left_idx, right_idx, prim_start, prim_count);
        
        float node_dist_sq = pointAABBDistanceSq(query, aabb_min, aabb_max);
        if (node_dist_sq > min_dist * min_dist) {
            continue;
        }
        
        if (prim_count > 0) {
            for (int i = 0; i < prim_count; i++) {
                int prim_idx = int(fetch_texel(prim_start + i, pc_bvh_prim_indices).x);
                vec3 point = fetch_point(prim_idx);
                float dist = length(query - point);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_idx = prim_idx;
                }
            }
        } else {
            vec3 left_aabb_min, left_aabb_max;
            int left_left, left_right, left_prim_start, left_prim_count;
            fetchBVHNode(pc_bvh_nodes, left_idx, left_aabb_min, left_aabb_max,
                        left_left, left_right, left_prim_start, left_prim_count);
            
            vec3 right_aabb_min, right_aabb_max;
            int right_left, right_right, right_prim_start, right_prim_count;
            fetchBVHNode(pc_bvh_nodes, right_idx, right_aabb_min, right_aabb_max,
                        right_left, right_right, right_prim_start, right_prim_count);
            
            float left_dist_sq = pointAABBDistanceSq(query, left_aabb_min, left_aabb_max);
            float right_dist_sq = pointAABBDistanceSq(query, right_aabb_min, right_aabb_max);
            
            if (left_dist_sq < right_dist_sq) {
                if (right_dist_sq <= min_dist * min_dist) stack[stack_ptr++] = right_idx;
                if (left_dist_sq <= min_dist * min_dist) stack[stack_ptr++] = left_idx;
            } else {
                if (left_dist_sq <= min_dist * min_dist) stack[stack_ptr++] = left_idx;
                if (right_dist_sq <= min_dist * min_dist) stack[stack_ptr++] = right_idx;
            }
        }
    }
    
    return closest_idx;
}

int bvhPointsInRadius(vec3 query, float radius, int max_results,
                      out int result_indices[max_neighborhood_size], out float result_dists[max_neighborhood_size]) {
    int count = 0;
    float radius_sq = radius * radius;
    
    int stack[stack_size];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;
    
    while (stack_ptr > 0 && count < max_results) {
        int node_idx = stack[--stack_ptr];
        
        vec3 aabb_min, aabb_max;
        int left_idx, right_idx, prim_start, prim_count;
        fetchBVHNode(pc_bvh_nodes, node_idx, aabb_min, aabb_max, left_idx, right_idx, prim_start, prim_count);
        
        float node_dist_sq = pointAABBDistanceSq(query, aabb_min, aabb_max);
        if (node_dist_sq > radius_sq) {
            continue;
        }
        
        if (prim_count > 0) {
            for (int i = 0; i < prim_count && count < max_results; i++) {
                int prim_idx = int(fetch_texel(prim_start + i, pc_bvh_prim_indices).x);
                vec3 point = fetch_point(prim_idx);
                float dist_sq = dot(query - point, query - point);
                if (dist_sq <= radius_sq) {
                    result_indices[count] = prim_idx;
                    result_dists[count] = sqrt(dist_sq);
                    count++;
                }
            }
        } else {
            stack[stack_ptr++] = left_idx;
            stack[stack_ptr++] = right_idx;
        }
    }
    
    return count;
}

// Returns actual number of neighbors found (may be less than k)
int bvhKNearestNeighbors(vec3 query, int k, int max_results,
                         out int result_indices[max_neighborhood_size], out float result_dists[max_neighborhood_size]) {
    if (n_pc_bvh_nodes == 0 || k <= 0) return 0;
    
    // Use max_results as the array size limit
    k = min(k, max_results);
    
    // Priority queue using simple insertion sort
    // We maintain the k closest points found so far
    int count = 0;
    float kth_dist_sq = 1e10;  // Distance to k-th nearest (infinity initially)
    
    int stack[stack_size];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;  // Start at root
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        
        vec3 aabb_min, aabb_max;
        int left_idx, right_idx, prim_start, prim_count;
        fetchBVHNode(pc_bvh_nodes, node_idx, aabb_min, aabb_max, left_idx, right_idx, 
                     prim_start, prim_count);
        
        // Prune: if closest point in AABB is farther than k-th nearest, skip
        float node_dist_sq = pointAABBDistanceSq(query, aabb_min, aabb_max);
        if (count >= k && node_dist_sq > kth_dist_sq) {
            continue;
        }
        
        // Leaf node: process points
        if (prim_count > 0) {
            for (int i = 0; i < prim_count; i++) {
                int prim_idx = int(fetch_texel(prim_start + i, pc_bvh_prim_indices).x);
                vec3 point = fetch_point(prim_idx);
                float dist_sq = dot(query - point, query - point);
                
                // Check if this point should be in the k-NN set
                if (count < k) {
                    // Haven't found k points yet, just add it
                    result_indices[count] = prim_idx;
                    result_dists[count] = sqrt(dist_sq);
                    count++;
                    
                    // Update kth_dist_sq (find maximum distance in current set)
                    kth_dist_sq = 0.0;
                    for (int j = 0; j < count; j++) {
                        kth_dist_sq = max(kth_dist_sq, result_dists[j] * result_dists[j]);
                    }
                } else if (dist_sq < kth_dist_sq) {
                    // Replace farthest point in k-NN set
                    // Find index of farthest point
                    int max_idx = 0;
                    float max_dist_sq = result_dists[0] * result_dists[0];
                    for (int j = 1; j < k; j++) {
                        float d_sq = result_dists[j] * result_dists[j];
                        if (d_sq > max_dist_sq) {
                            max_dist_sq = d_sq;
                            max_idx = j;
                        }
                    }
                    
                    // Replace
                    result_indices[max_idx] = prim_idx;
                    result_dists[max_idx] = sqrt(dist_sq);
                    
                    // Update kth_dist_sq
                    kth_dist_sq = 0.0;
                    for (int j = 0; j < k; j++) {
                        kth_dist_sq = max(kth_dist_sq, result_dists[j] * result_dists[j]);
                    }
                }
            }
        } else {
            // Internal node: push children
            // Prioritize closer child for better pruning
            vec3 left_aabb_min, left_aabb_max;
            int left_left, left_right, left_prim_start, left_prim_count;
            fetchBVHNode(pc_bvh_nodes, left_idx, left_aabb_min, left_aabb_max,
                        left_left, left_right, left_prim_start, left_prim_count);
            
            vec3 right_aabb_min, right_aabb_max;
            int right_left, right_right, right_prim_start, right_prim_count;
            fetchBVHNode(pc_bvh_nodes, right_idx, right_aabb_min, right_aabb_max,
                        right_left, right_right, right_prim_start, right_prim_count);
            
            float left_dist_sq = pointAABBDistanceSq(query, left_aabb_min, left_aabb_max);
            float right_dist_sq = pointAABBDistanceSq(query, right_aabb_min, right_aabb_max);
            
            // Push farther child first (so closer is processed first)
            if (left_dist_sq < right_dist_sq) {
                if (count < k || right_dist_sq <= kth_dist_sq) stack[stack_ptr++] = right_idx;
                if (count < k || left_dist_sq <= kth_dist_sq) stack[stack_ptr++] = left_idx;
            } else {
                if (count < k || left_dist_sq <= kth_dist_sq) stack[stack_ptr++] = left_idx;
                if (count < k || right_dist_sq <= kth_dist_sq) stack[stack_ptr++] = right_idx;
            }
        }
    }
    
    return min(count, k);
}

float dot2( in vec3 v ) { return dot(v,v); }

vec3 closestPointOnTriangle(vec3 p, vec3 v1, vec3 v2, vec3 v3, out vec3 normal) {
    vec3 v21 = v2 - v1; vec3 p1 = p - v1;
    vec3 v32 = v3 - v2; vec3 p2 = p - v2;
    vec3 v13 = v1 - v3; vec3 p3 = p - v3;
    vec3 nor = cross(v21, v13);
    normal = -nor;
    
    // inside/outside test
    bool isInside = (sign(dot(cross(v21, nor), p1)) + 
                     sign(dot(cross(v32, nor), p2)) + 
                     sign(dot(cross(v13, nor), p3)) >= 2.0);
    
    if (isInside) {
        // Point projects inside triangle - closest point is on face
        float t = dot(nor, p1) / dot2(nor);
        return p - t * nor;
    } else {
        // Point projects outside - find closest point on edges
        float t1 = clamp(dot(v21, p1) / dot2(v21), 0.0, 1.0);
        float t2 = clamp(dot(v32, p2) / dot2(v32), 0.0, 1.0);
        float t3 = clamp(dot(v13, p3) / dot2(v13), 0.0, 1.0);
        
        vec3 closest1 = p1 - v21 * t1;
        vec3 closest2 = p2 - v32 * t2;
        vec3 closest3 = p3 - v13 * t3;
        
        float d1 = dot2(closest1);
        float d2 = dot2(closest2);
        float d3 = dot2(closest3);
        
        // Return closest of the three edge points
        if (d1 < d2 && d1 < d3) return v1 + v21 * t1;
        if (d2 < d3) return v2 + v32 * t2;
        return v3 + v13 * t3;
    }
}

float pointTriangleDistanceSq(vec3 point, vec3 v0, vec3 v1, vec3 v2) {
    vec3 normal_dummy;
    vec3 closest = closestPointOnTriangle(point, v0, v1, v2, normal_dummy);
    vec3 diff = point - closest;
    return dot(diff, diff);
}

int bvhKNearestTriangles(sampler2D mesh, sampler2D bvh_nodes, sampler2D bvh_indices, 
                        int n_bvh_nodes, vec3 query, int k, int max_results,
                        out int result_indices[max_neighborhood_size], out float result_dists[max_neighborhood_size]) {
    if (n_bvh_nodes == 0) {
        return 0;
    }
    
    int count = 0;
    
    // Initialize result arrays with large distances
    for (int i = 0; i < max_results; i++) {
        result_dists[i] = INFINITY;
        result_indices[i] = -1;
    }
    
    int stack[stack_size];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        
        vec3 aabb_min, aabb_max;
        int left_idx, right_idx, prim_start, prim_count;
        fetchBVHNode(bvh_nodes, node_idx, aabb_min, aabb_max, left_idx, right_idx, prim_start, prim_count);
        
        // Check if this node could contain closer triangles than our current k-th nearest
        float node_dist_sq = pointAABBDistanceSq(query, aabb_min, aabb_max);
        float kth_dist_sq = (count >= k) ? result_dists[k-1] * result_dists[k-1] : INFINITY;
        
        if (node_dist_sq > kth_dist_sq) {
            continue;
        }
        
        if (prim_count > 0) {
            // Leaf node: check all triangles
            for (int i = 0; i < prim_count && count < max_results; i++) {
                int tri_idx = int(fetch_texel(prim_start + i, bvh_indices).x);
                
                // Fetch triangle vertices
                vec3 v0 = fetch_texel(3 * tri_idx, mesh).xyz;
                vec3 v1 = fetch_texel(3 * tri_idx + 1, mesh).xyz;
                vec3 v2 = fetch_texel(3 * tri_idx + 2, mesh).xyz;
                
                // Compute distance to triangle
                float dist_sq = pointTriangleDistanceSq(query, v0, v1, v2);
                float dist = sqrt(dist_sq);
                
                // Insert into sorted list if closer than k-th element
                if (count < k || dist < result_dists[k-1]) {
                    // Find insertion position
                    int insert_pos = count;
                    for (int j = 0; j < count; j++) {
                        if (dist < result_dists[j]) {
                            insert_pos = j;
                            break;
                        }
                    }
                    
                    // Shift elements right
                    int shift_end = min(count, k - 1);
                    for (int j = shift_end; j > insert_pos; j--) {
                        result_dists[j] = result_dists[j - 1];
                        result_indices[j] = result_indices[j - 1];
                    }
                    
                    // Insert new element
                    result_dists[insert_pos] = dist;
                    result_indices[insert_pos] = tri_idx;
                    
                    if (count < k) count++;
                }
            }
        } else {
            // Internal node: push children based on distance
            vec3 left_aabb_min, left_aabb_max;
            int left_left, left_right, left_prim_start, left_prim_count;
            fetchBVHNode(bvh_nodes, left_idx, left_aabb_min, left_aabb_max,
                        left_left, left_right, left_prim_start, left_prim_count);
            
            vec3 right_aabb_min, right_aabb_max;
            int right_left, right_right, right_prim_start, right_prim_count;
            fetchBVHNode(bvh_nodes, right_idx, right_aabb_min, right_aabb_max,
                        right_left, right_right, right_prim_start, right_prim_count);
            
            float left_dist_sq = pointAABBDistanceSq(query, left_aabb_min, left_aabb_max);
            float right_dist_sq = pointAABBDistanceSq(query, right_aabb_min, right_aabb_max);
            
            // Push farther child first (so closer child is processed first)
            if (left_dist_sq < right_dist_sq) {
                if (right_dist_sq <= kth_dist_sq) stack[stack_ptr++] = right_idx;
                if (left_dist_sq <= kth_dist_sq) stack[stack_ptr++] = left_idx;
            } else {
                if (left_dist_sq <= kth_dist_sq) stack[stack_ptr++] = left_idx;
                if (right_dist_sq <= kth_dist_sq) stack[stack_ptr++] = right_idx;
            }
        }
    }
    
    return count;
}

bool intersectSphere( in vec3 ro, in vec3 rd, in vec3 p, in float r, out float t, out vec3 n ) {
    t = dot( rd, p-ro ) / dot( rd, rd );
    float d = length( ro + t * rd - p );
    
    if ( d > r ) return false;
    
    float s = sqrt( r * r - d * d );
    if ( t - s >= 0. ) {
        t -= s;
    } else {
        t += s;
    }
    
    n = normalize( ro + t * rd - p );
    
    return t > 0.;
}

bool bvhIntersectPointCloud( in vec3 ro, in vec3 rd, in float t_min, in float t_max,
                            in float sphere_radius,
                            out float hit_t, out int hit_point_idx, out vec3 hit_normal) {
    if (n_pc_bvh_nodes == 0) {
        return false;  // No BVH available
    }
    
    hit_t = t_max;
    hit_point_idx = -1;
    bool found_hit = false;
    
    int stack[stack_size];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;  // Start at root
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        
        // Fetch node
        vec3 aabb_min, aabb_max;
        int left_idx, right_idx, prim_start, prim_count;
        fetchBVHNode(pc_bvh_nodes, node_idx, aabb_min, aabb_max, left_idx, right_idx, prim_start, prim_count);

        vec3 expansion = vec3(sphere_radius);
        aabb_min -= expansion;
        aabb_max += expansion;
        
        // Test AABB intersection
        float t_near, t_far;
        if (!intersectAABB(ro, rd, aabb_min, aabb_max, t_near, t_far)) {
            continue;
        }
        
        // Skip if AABB is behind current closest hit
        if (t_near > hit_t || t_far < t_min) {
            continue;
        }
        
        // Leaf node: test point spheres
        if (prim_count > 0) {
            for (int i = 0; i < prim_count; i++) {
                int point_idx = int(fetch_texel(prim_start + i, pc_bvh_prim_indices).x);
                vec3 point_pos = fetch_point(point_idx);
                
                // Test ray-sphere intersection (using existing function)
                float t;
                vec3 n;
                if (intersectSphere(ro, rd, point_pos, sphere_radius, t, n)) {
                    if (t >= t_min && t < hit_t) {
                        hit_t = t;
                        hit_point_idx = point_idx;
                        hit_normal = n;
                        found_hit = true;
                    }
                }
            }
        } else {
            // Internal node: push children
            stack[stack_ptr++] = left_idx;
            stack[stack_ptr++] = right_idx;
        }
    }
    
    return found_hit;
}

bool intersectTriangle(in vec3 ro, in vec3 rd, in vec3 v0, in vec3 v1, in vec3 v2,
                       out float t, out vec2 bary) {
    // Möller-Trumbore algorithm
    const float EPSILON = 1e-8;
    
    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    vec3 h = cross(rd, edge2);
    float a = dot(edge1, h);
    
    // Ray parallel to triangle
    if (abs(a) < EPSILON) return false;
    
    float f = 1.0 / a;
    vec3 s = ro - v0;
    float u = f * dot(s, h);
    
    if (u < 0.0 || u > 1.0) return false;
    
    vec3 q = cross(s, edge1);
    float v = f * dot(rd, q);
    
    if (v < 0.0 || u + v > 1.0) return false;
    
    t = f * dot(edge2, q);
    
    if (t < EPSILON) return false;
    
    bary = vec2(u, v);
    return true;
}

bool bvhIntersectMesh(in sampler2D mesh, in sampler2D bvh_nodes, in sampler2D bvh_indices, in int n_bvh_nodes, in vec3 ro, in vec3 rd, in float t_min, in float t_max,
                      out float hit_t, out int hit_tri_idx, out vec2 hit_bary) {
    if (n_bvh_nodes == 0) {
        return false;  // No BVH available
    }
    
    hit_t = t_max;
    hit_tri_idx = -1;
    bool found_hit = false;
    
    int stack[stack_size];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;  // Start at root
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        
        // Fetch node from mesh BVH
        int base_idx = node_idx * 3;
        vec4 data0 = fetch_texel(base_idx, bvh_nodes);
        vec4 data1 = fetch_texel(base_idx + 1, bvh_nodes);
        vec4 data2 = fetch_texel(base_idx + 2, bvh_nodes);
        
        vec3 aabb_min = data0.xyz;
        vec3 aabb_max = vec3(data0.w, data1.xy);
        int left_idx = int(data1.z);
        int right_idx = int(data1.w);
        int prim_start = int(data2.x);
        int prim_count = int(data2.y);
        
        // Test AABB intersection
        float t_near, t_far;
        if (!intersectAABB(ro, rd, aabb_min, aabb_max, t_near, t_far)) {
            continue;
        }
        
        // Skip if AABB is behind current closest hit
        if (t_near > hit_t || t_far < t_min) {
            continue;
        }
        
        // Leaf node: test triangles
        if (prim_count > 0) {
            for (int i = 0; i < prim_count; i++) {
                int tri_idx = int(fetch_texel(prim_start + i, bvh_indices).x);
                
                // Fetch triangle vertices from shape texture
                vec3 v0 = fetch_texel(3 * tri_idx, mesh).xyz;
                vec3 v1 = fetch_texel(3 * tri_idx + 1, mesh).xyz;
                vec3 v2 = fetch_texel(3 * tri_idx + 2, mesh).xyz;
                
                // Test ray-triangle intersection
                float t;
                vec2 bary;
                if (intersectTriangle(ro, rd, v0, v1, v2, t, bary)) {
                    if (t >= t_min && t < hit_t) {
                        hit_t = t;
                        hit_tri_idx = tri_idx;
                        hit_bary = bary;
                        found_hit = true;
                    }
                }
            }
        } else {
            // Internal node: push children
            // Push in order (closer first for better early termination)
            stack[stack_ptr++] = left_idx;
            stack[stack_ptr++] = right_idx;
        }
    }
    
    return found_hit;
}

// ==== colormaps / visualization

/* shadertoy.com/view/tstcDX */ vec3 rgb2hsv(vec3 c); vec3 hsv2rgb(vec3 c); vec3 rgb2hsv(vec3 c){const vec4 K=vec4(0.,-1./3.,2./3.,-1.);vec4 p=mix(vec4(c.bg,K.wz),vec4(c.gb,K.xy),step(c.b,c.g));vec4 q=mix(vec4(p.xyw,c.r),vec4(c.r,p.yzx),step(p.x,c.r));float d=q.x-min(q.w,q.y);const float e=1.e-10;return vec3(abs(q.z+(q.w-q.y)/(6.*d+e)),d/(q.x+e),q.x);} vec3 hsv2rgb(vec3 c){const vec4 K=vec4(1.,2./3.,1./3.,3.);vec3 p=abs(fract(c.xxx+K.xyz)*6.-K.www);return c.z*mix(K.xxx,clamp(p-K.xxx,0.,1.),c.y);}

// return the index of the closest point in the point cloud
int closestPoint(vec3 q) {
    if (n_pc_bvh_nodes == 0) {
        // Fallback to brute force
        float d = INFINITY;
        int idx = 0;
        for (int i = 0; i < n_points; i++) {
            float dist = length(q - fetch_point(i));
            if (dist < d) {
                d = dist;
                idx = i;
            }
        }
        return idx;
    }
    
    // Use BVH
    float min_dist;
    int idx = bvhClosestPoint(q, min_dist);
    return idx;
}

// return the index of the farthest point in the point cloud
int farthest_point(vec3 q) {
    if (n_pc_bvh_nodes == 0 || neighborhood_size <= 0 || neighborhood_size >= n_points) {
        // Fallback
        float d = 0;
        int idx = 0;
        for (int i = 0; i < n_points; i++) {
            float dist = length(q - fetch_point(i));
            if (dist > d) {
                d = dist;
                idx = i;
            }
        }
        return idx;
    }
    
    // Use BVH
    int indices[max_neighborhood_size];
    float dists[max_neighborhood_size];
    int count = bvhPointsInRadius(q, neighborhood_radius, max_neighborhood_size, indices, dists);
    if (count == 0) {
        count = bvhKNearestNeighbors(q, neighborhood_size, max_neighborhood_size, indices, dists);
    }
    float max_dist = 0.0;
    int idx = indices[0];
    for (int i = 0; i < count; i++) {
        if (dists[i] > max_dist) {
            max_dist = dists[i];
            idx = indices[i];
        }
    }
    return idx;
}

float compute_shift(in vec3 q) {
    return 0.5*length(q-fetch_point(farthest_point(q)));
}

float segment(float edge0, float edge1, float x) {
    return step(edge0,x) * (1.0-step(edge1,x));
}

float normalize_scalar(float minVal, float maxVal, float val) {
    return (val - minVal) / (maxVal - minVal);
}

vec3 colormap_PasadenaBlue(float t) {
    vec3 blue = vec3(0.0627451, 0.517647, 0.94902);
    vec3 white = vec3(1.0, 1.0, 1.0);
    float tp = pow(t,7.0);
    return (1.0-tp)*white + tp*blue;
}

vec3 colormap_grapefruit(float t) {
    vec3 grapefruit = vec3(0.992157, 0.431373, 0.337255);
    vec3 white = vec3(1.0, 1.0, 1.0);
    return mix(white, grapefruit, t);
}

vec3 colormap_SDF(float t) {
    return segment(0.0,0.5,t) * colormap_PasadenaBlue(2.0*(t-0.0)) +
           segment(0.5,1.0,t) * colormap_grapefruit(2.0*(t-0.5));
}

/*
 * Helper function for darken_color()
 * https://stackoverflow.com/a/141943
 */
vec3 redistribute_rgb(vec3 color) {
    
    float r = color.x; float g = color.y; float b = color.z;
    // If no need for clamping
    float m = max(max(r,g),b);
    if (m <= 1.0) {
        return color;
    }
    // If all values exceed max
    float total = r + g + b;
    if (total >= 3.0) {
        return vec3(1.0, 1.0, 1.0);
    }
    // Else, "redistribute" values so the hue is preserved.
    float x = (3.0 - total) / (3.0 * m - total);
    float gray = 1.0 - x * m;
    return vec3(gray + x * r, gray + x * g, gray + x * b);
}

/*
 * Darken a color, with 0 <= t <= 1.
 * t = 0 corresponds to black, t = 1 corresponds to original color.
 */
vec3 darken_color(vec3 color, float t) {
    
    return redistribute_rgb(t * color);
}

//==== convolutional SDF approximations

/* unsigned distance */
float logSumExp(in vec3 q, in float lambda_scale) {
    float shift = compute_shift(q);
    float lambda = lambda_scale * maxExpArg / shift;
    shift = maxExpArg/lambda;
    float w = 0.;

    for ( int i = 0; i < n_points; i++ ) {
        vec3 rVec = q - fetch_point(i);
        float r = length(rVec);
        float weight = exp(-lambda*(r-shift));
        w += weight;
    }
    return -log(w)/lambda + shift;
}

vec3 logSumExpGradient(in vec3 q, in float lambda_scale) {
    vec3 A = vec3(0.,0.,0.);
    vec3 B = vec3(0.,0.,0.);
    float C = 0.;
    float shift = compute_shift(q);
    float lambda = lambda_scale * maxExpArg / shift;
    for ( int i = 0; i < n_points; i++ ) {
        vec3 rVec = q - fetch_point(i);
        float r = length(rVec);
        vec3 rHat = rVec/r;
        float weight = exp(-lambda*(r-shift));
        float h = 1.;
        vec3 grad_h = vec3(0.,0.,0.);
        A += grad_h*weight;
        B += lambda*rHat*h*weight;
        C += weight*h;
    }
    return -(A-B)/(C*lambda);
}

void logSumExpAndGradient(in vec3 q, in float lambda_scale, out float phi, out vec3 gradient) {
    vec3 A = vec3(0.,0.,0.);
    vec3 B = vec3(0.,0.,0.);
    float C = 0.;
    float shift = compute_shift(q);
    float lambda = lambda_scale * maxExpArg / shift;
    float w = 0.;
    for ( int i = 0; i < n_points; i++ ) {
        vec3 rVec = q - fetch_point(i);
        float r = length(rVec);
        vec3 rHat = rVec/r;
        float weight = exp(-lambda*(r-shift));
        w += weight;
        float h = 1.;
        vec3 grad_h = vec3(0.,0.,0.);
        A += grad_h*weight;
        B += lambda*rHat*h*weight;
        C += weight*h;
    }
    gradient = -(A-B)/(C*lambda);
    phi = -log(w)/lambda + shift;
}

/* unsigned distance */
float selfNormalizedLogSumExp(in vec3 q, in float lambda_scale) {
    float u = 0.;
    float normalization = 0.;
    float shift = compute_shift(q);
    float lambda = lambda_scale * maxExpArg / shift;
    for ( int i = 0; i < n_points; i++ ) {
        vec3 rVec = q - fetch_point(i);
        float r = length(rVec);
        float weight = exp(-lambda*(r-shift));
        if (usePointAreas) {
            float area = fetch_texel(2*i, pointcloud).w;
            weight *= area;
        }
        u += r * weight;
        normalization += weight;
    }
    return u/normalization;
}

vec3 selfNormalizedLogSumExpGradient(in vec3 q, in float lambda_scale) {
    vec3 A = vec3(0.,0.,0.);
    vec3 B = vec3(0.,0.,0.);
    float C = 0.;
    float D = 0.;
    float shift = compute_shift(q);
    float lambda = lambda_scale * maxExpArg / shift;
    for ( int i = 0; i < n_points; i++ ) {
        vec3 p_i = fetch_point(i);
        vec3 rVec = q - p_i;
        float r = length(rVec);
        vec3 rHat = rVec/r;
        float weight = exp(-lambda*(r-shift));
        if (usePointAreas) {
            float area = fetch_texel(2*i, pointcloud).w;
            weight *= area;
        }
        float h = r;
        vec3 grad_h = rHat;
        A += (grad_h - lambda*h*rHat)*weight;
        B += rHat*weight;
        C += h*weight;
        D += weight;
    }
    return A/D + lambda*(B*C)/(D*D);
}

void selfNormalizedLogSumExpAndGradient(in vec3 q, in float lambda_scale, out float phi, out vec3 gradient) {
    vec3 A = vec3(0.,0.,0.);
    vec3 B = vec3(0.,0.,0.);
    float C = 0.;
    float D = 0.;
    float u = 0.;
    float normalization = 0.;
    float shift = compute_shift(q);
    float lambda = lambda_scale * maxExpArg / shift;
    for ( int i = 0; i < n_points; i++ ) {
        vec3 p_i = fetch_point(i);
        vec3 rVec = q - p_i;
        float r = length(rVec);
        vec3 rHat = rVec/r;
        float weight = exp(-lambda*(r-shift));
        if (usePointAreas) {
            float area = fetch_texel(2*i, pointcloud).w;
            weight *= area;
        }
        float h = r;
        vec3 grad_h = rHat;
        u += h * weight;
        normalization += weight;
        A += (grad_h - lambda*h*rHat)*weight;
        B += rHat*weight;
        C += h*weight;
        D += weight;
    }
    phi = u/normalization;
    gradient = A/D + lambda*(B*C)/(D*D);
}

void windingNumber(in vec3 q, out float phi) {
    phi = 0.0;
    for ( int i = 0; i < n_points; i++ ) {
        vec3 p_i = fetch_point(i);
        vec3 n_i = fetch_normal(i);
        vec3 r_vec = q - p_i;
        float r = length(r_vec);
        float P = dot(n_i, r_vec) / (r*r*r) / (4.0 * PI);
        float area = fetch_texel(2*i, pointcloud).w; // warning: sometimes texture is overloaded to vis outliers
        phi += area * P;
    }
}

float signedLogSumExpGradient(in vec3 q, in float lambda_scale, out vec3 gradient) {
    
    float shift = compute_shift(q);
    float lambda = lambda_scale * maxExpArg / shift;

    if (n_pc_bvh_nodes == 0 || neighborhood_size <= 0 || neighborhood_size >= n_points) {
        vec3 A = vec3(0.,0.,0.);
        vec3 B = vec3(0.,0.,0.);
        float C = 0.;
        float w = 0.;
        for ( int i = 0; i < n_points; i++ ) {
            vec3 p_i = fetch_point(i);
            vec3 rVec = q - p_i;
            float r = length(rVec);
            float weight = exp(-lambda*(r - shift));
                if (usePointAreas) {
                float area = fetch_texel(2*i, pointcloud).w;
                weight *= area;
            }
            vec3 n_i = fetch_normal(i);
            //float h = dot(rVec, n_i)/r;
            float h = (lambda * r + 1.0) * dot(rVec, n_i) / (2.*PI*r*r*r);
            vec3 rHat = rVec/r;
            vec3 grad_h = n_i/dot(rVec,n_i) - rVec/dot(rVec,rVec);
            w += h * weight;
            A += grad_h*weight;
            B += lambda*rHat*h*weight;
            C += weight*h;
        }
        gradient = -(A-B)/(C*lambda);
        return sign(w)*(-log(abs(w))/lambda + shift);
    }

    // BVH
    int indices[max_neighborhood_size];
    float dists[max_neighborhood_size];
    int count = bvhKNearestNeighbors(q, neighborhood_size, max_neighborhood_size, indices, dists);
    if (count == 0) return 0.0;

    float w = 0.;
    vec3 A = vec3(0.,0.,0.);
    vec3 B = vec3(0.,0.,0.);
    float C = 0.;
    for ( int idx = 0; idx < count; idx++ ) {
        int i = indices[idx];
        vec3 p_i = fetch_point(i);
        vec3 rVec = q - p_i;
        float r = length(rVec);
        float weight = exp(-lambda*(r-shift));
        if (usePointAreas) {
            float area = fetch_texel(2*i, pointcloud).w;
            weight *= area;
        }
        vec3 n_i = fetch_normal(i);
        float h = dot(rVec, n_i)/r;
        vec3 rHat = rVec/r;
        vec3 grad_h = n_i/dot(rVec,n_i) - rVec/dot(rVec,rVec);
        w += h * weight;
        A += grad_h*weight;
        B += lambda*rHat*h*weight;
        C += weight*h;
    }
    gradient = -(A-B)/(C*lambda);
    return -sign(w)*(log(abs(w))/lambda);
}

float regularizedSignedLogSumExpGradient(in vec3 q, in float lambda_scale, out vec3 gradient) {
    
    float shift = compute_shift(q);
    float lambda = lambda_scale * maxExpArg / shift;

    if (n_pc_bvh_nodes == 0 || neighborhood_size <= 0 || neighborhood_size >= n_points) {
        vec3 A = vec3(0.,0.,0.);
        vec3 B = vec3(0.,0.,0.);
        float C = 0.;
        float w = 0.;
        for ( int i = 0; i < n_points; i++ ) {
            vec3 p_i = fetch_point(i);
            vec3 rVec = q - p_i;
            float r = length(rVec);
            float t = r / reg_epsilon;
            float S = erf(t) - 2.0*t / sqrt(PI) * exp(-t*t);
            float weight = exp(-lambda*(r - shift));
            if (usePointAreas) {
                float area = fetch_texel(2*i, pointcloud).w;
                weight *= area;
            }
            vec3 n_i = fetch_normal(i);
            //float h = dot(rVec, n_i)/r;
            float h = (lambda * r + 1.0) * dot(rVec, n_i) / (2.*PI*r*r*r);
            vec3 rHat = rVec/r;
            vec3 grad_h = n_i/dot(rVec,n_i) - rVec/dot(rVec,rVec);
            w += S * h * weight;
            A += grad_h*weight;
            B += lambda*rHat*h*weight;
            C += weight*h;
        }
        gradient = -(A-B)/(C*lambda);
        return sign(w)*(-log(abs(w))/lambda + shift);
    }

    // BVH
    int indices[max_neighborhood_size];
    float dists[max_neighborhood_size];
    int count = bvhKNearestNeighbors(q, neighborhood_size, max_neighborhood_size, indices, dists);
    if (count == 0) return 0.0;

    float w = 0.;
    vec3 A = vec3(0.,0.,0.);
    vec3 B = vec3(0.,0.,0.);
    float C = 0.;
    for ( int idx = 0; idx < count; idx++ ) {
        int i = indices[idx];
        vec3 p_i = fetch_point(i);
        vec3 rVec = q - p_i;
        float r = length(rVec);
        float t = r / reg_epsilon;
        float S = erf(t) - 2.0*t / sqrt(PI) * exp(-t*t);
        float weight = exp(-lambda*(r-shift));
        if (usePointAreas) {
            float area = fetch_texel(2*i, pointcloud).w;
            weight *= area;
        }
        vec3 n_i = fetch_normal(i);
        float h = dot(rVec, n_i)/r;
        vec3 rHat = rVec/r;
        vec3 grad_h = n_i/dot(rVec,n_i) - rVec/dot(rVec,rVec);
        w += S * h * weight;
        A += grad_h*weight;
        B += lambda*rHat*h*weight;
        C += weight*h;
    }
    gradient = -(A-B)/(C*lambda);
    return -sign(w)*(log(abs(w))/lambda);
}

float selfNormalizedSignedLogSumExpGradient(in vec3 q, in float lambda_scale, out vec3 gradient) {
    
    float shift = compute_shift(q);
    float lambda = lambda_scale * maxExpArg / shift;

    if (n_pc_bvh_nodes == 0 || neighborhood_size <= 0 || neighborhood_size >= n_points) {
        float u = 0.;
        vec3 A = vec3(0.,0.,0.);
        vec3 B = vec3(0.,0.,0.);
        float C = 0.;
        float D = 0.;
        for ( int i = 0; i < n_points; i++ ) {
            vec3 p_i = fetch_point(i);
            vec3 rVec = q - p_i;
            float r = length(rVec);
            vec3 rHat = rVec / r;
            float weight = exp(-lambda*(r-shift));
            if (usePointAreas) {
                float area = fetch_texel(2*i, pointcloud).w;
                weight *= area;
            }
            vec3 n_i = fetch_normal(i);
            float h = dot(rVec, n_i);
            vec3 grad_h = n_i;
            u += dot(rVec, fetch_normal(i)) * weight;
            A += (grad_h - lambda*h*rHat)*weight;
            B += rHat*weight;
            C += h*weight;
            D += weight;
        }
        gradient = A/D + lambda*(B*C)/(D*D);
        return u / D;
    }

    // BVH
    int indices[max_neighborhood_size];
    float dists[max_neighborhood_size];
    int count = bvhKNearestNeighbors(q, neighborhood_size, max_neighborhood_size, indices, dists);
    if (count == 0) return 0.;

    vec3 A = vec3(0.,0.,0.);
    vec3 B = vec3(0.,0.,0.);
    float C = 0.;
    float D = 0.;
    float u = 0.;
    for ( int idx = 0; idx < count; idx++ ) {
        int i = indices[idx];
        vec3 p_i = fetch_point(i);
        vec3 rVec = q - p_i;
        float r = length(rVec);
        vec3 rHat = rVec / r;
        float weight = exp(-lambda*(r-shift));
        if (usePointAreas) {
            float area = fetch_texel(2*i, pointcloud).w;
            weight *= area;
        }
        vec3 n_i = fetch_normal(i);
        float h = dot(rVec, n_i);
        vec3 grad_h = n_i;
        u += dot(rVec, fetch_normal(i)) * weight;
        A += (grad_h - lambda*h*rHat)*weight;
        B += rHat*weight;
        C += h*weight;
        D += weight;
    }
    gradient = A/D + lambda*(B*C)/(D*D);
    return u/D;
}

float signedTorusDistance(vec3 q, vec3 center, vec3 axis, float R, float r) {
    vec3 rVec = q - center;
    vec2 u = vec2(length(cross(rVec, axis)) - R, dot(rVec, axis));
    float d = length(u) - abs(r);
    return sign(r) * d;

    vec3 rad = rVec - dot(rVec, axis) * axis;
    vec2 v = vec2(length(rad) - R, dot(rVec, axis));
    v /= length(v);
    vec3 cp = center + (R + r * v.x) * normalize(rad) + r * v.y * axis; // closest point
}

vec3 rotation_to_cartesian(in vec2 angles) {
  float theta = angles.x;
  float phi = angles.y;
  float sin_phi = sin(phi);
  return vec3(cos(theta) * sin_phi, sin(theta) * sin_phi, cos(phi));
}

void get_torus_params(in int i, in sampler2D texture, out vec3 center, out vec3 axis, out float R, out float r) {
    center = fetch_texel(2*i, texture).xyz;
    vec2 axis2d = fetch_texel(2*i+1, texture).xy;
    axis = rotation_to_cartesian(axis2d);
    vec2 radii = fetch_texel(2*i+1, texture).zw;
    R = radii.x; r = radii.y;
}

// BVH traversal

struct RadiusStats {
    int count;
    float max_dist;
    float sum_dist;
};

void bvhGatherStats(vec3 query, float radius, out RadiusStats stats) {
    stats.count = 0;
    stats.max_dist = 0.0;
    stats.sum_dist = 0.0;
    
    if (n_pc_bvh_nodes == 0) return;
    
    float radius_sq = radius * radius;
    
    int stack[stack_size];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        
        vec3 aabb_min, aabb_max;
        int left_idx, right_idx, prim_start, prim_count;
        fetchBVHNode(pc_bvh_nodes, node_idx, aabb_min, aabb_max, 
                     left_idx, right_idx, prim_start, prim_count);
        
        float node_dist_sq = pointAABBDistanceSq(query, aabb_min, aabb_max);
        if (node_dist_sq > radius_sq) {
            continue;
        }
        
        if (prim_count > 0) {
            // Leaf node: process points
            for (int i = 0; i < prim_count; i++) {
                int prim_idx = int(fetch_texel(prim_start + i, pc_bvh_prim_indices).x);
                vec3 point = fetch_point(prim_idx);
                vec3 diff = query - point;
                float dist_sq = dot(diff, diff);
                
                if (dist_sq <= radius_sq) {
                    float dist = sqrt(dist_sq);
                    stats.count++;
                    stats.sum_dist += dist;
                    stats.max_dist = max(stats.max_dist, dist);
                }
            }
        } else {
            // Internal node: push children
            stack[stack_ptr++] = left_idx;
            stack[stack_ptr++] = right_idx;
        }
    }
}

// ----------------------------------------------------------------------------
// PASS 2: Weighted accumulation
// ----------------------------------------------------------------------------

struct TorusAccumulator {
    float g;                        // weighted sum of g_i(x)
    vec3 grad_g_minus_lambda_n_g;   // weighted sum of (∇g_i - λ r̂_i g_i)
    vec3 n;                         // weighted sum of r̂_i
    float normalization;            // sum of weights
};

// Accumulate contribution from a single point
void accumulateTorusPoint(vec3 q, int pointIdx, sampler2D torusTexture,
                          float lambda, float shift, inout TorusAccumulator acc) {
    vec3 p_i = fetch_point(pointIdx);
    vec3 diff = q - p_i;
    float dist = length(diff);
    
    // Skip if point is exactly at query (avoid division by zero)
    if (dist < 1e-10) return;
    
    float weight = exp(-lambda * (dist - shift));
    
    // Skip if weight is NaN or negligible
    if (isnan(weight) || weight < 1e-30) return;
    
    // Get torus parameters
    vec3 center, axis;
    float R, r;
    get_torus_params(pointIdx, torusTexture, center, axis, R, r);
    
    // Compute torus signed distance g_i(q)
    vec3 rVec = q - center;
    float v = length(cross(rVec, axis));
    
    // Skip degenerate case
    if (v < 1e-10) return;
    
    vec2 u = vec2(v - R, dot(rVec, axis));
    float u_len = length(u);
    if (u_len < 1e-10) return;
    
    float d = u_len - abs(r);
    float s = sign(r);
    float g_z = s * d;
    
    if (isnan(g_z)) return;
    
    // Compute gradient ∇g_i(q)
    // G is the 2x3 Jacobian of (v - R, dot(rVec, axis)) w.r.t. q
    float G_11 = (axis.x*axis.y*(center.y-q.y) + axis.x*axis.z*(center.z-q.z) 
                  - (axis.y*axis.y + axis.z*axis.z)*(center.x-q.x)) / v;
    float G_12 = (axis.x*axis.y*(center.x-q.x) + axis.y*axis.z*(center.z-q.z) 
                  - (axis.x*axis.x + axis.z*axis.z)*(center.y-q.y)) / v;
    float G_13 = (axis.x*axis.z*(center.x-q.x) + axis.y*axis.z*(center.y-q.y) 
                  - (axis.x*axis.x + axis.y*axis.y)*(center.z-q.z)) / v;
    float G_21 = axis.x;
    float G_22 = axis.y;
    float G_23 = axis.z;
    
    vec2 w = u / u_len;
    vec3 grad_g = s * vec3(w.x*G_11 + w.y*G_21, 
                           w.x*G_12 + w.y*G_22, 
                           w.x*G_13 + w.y*G_23);
    
    // Unit direction from point to query
    vec3 rHat = diff / dist;
    
    // Accumulate
    acc.grad_g_minus_lambda_n_g += weight * (grad_g - lambda * rHat * g_z);
    acc.g += weight * g_z;
    acc.n += weight * rHat;
    acc.normalization += weight;
}

// BVH traversal with direct accumulation
void bvhAccumulateInRadius(vec3 query, float radius, sampler2D torusTexture,
                           float lambda, float shift, inout TorusAccumulator acc) {
    if (n_pc_bvh_nodes == 0) return;
    
    float radius_sq = radius * radius;
    
    int stack[stack_size];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        
        vec3 aabb_min, aabb_max;
        int left_idx, right_idx, prim_start, prim_count;
        fetchBVHNode(pc_bvh_nodes, node_idx, aabb_min, aabb_max,
                     left_idx, right_idx, prim_start, prim_count);
        
        float node_dist_sq = pointAABBDistanceSq(query, aabb_min, aabb_max);
        if (node_dist_sq > radius_sq) {
            continue;
        }
        
        if (prim_count > 0) {
            // Leaf node: accumulate all points in radius
            for (int i = 0; i < prim_count; i++) {
                int prim_idx = int(fetch_texel(prim_start + i, pc_bvh_prim_indices).x);
                vec3 point = fetch_point(prim_idx);
                float dist_sq = dot(query - point, query - point);
                
                if (dist_sq <= radius_sq) {
                    accumulateTorusPoint(query, prim_idx, torusTexture, 
                                         lambda, shift, acc);
                }
            }
        } else {
            // Internal node: push children
            stack[stack_ptr++] = left_idx;
            stack[stack_ptr++] = right_idx;
        }
    }
}

float blendedTorusDistanceGradient(in vec3 q, in sampler2D texture, 
                                   in float lambda_scale, out vec3 gradient) {
    
    // Handle case where BVH is not available or we want all points
    if (n_pc_bvh_nodes == 0 || neighborhood_size <= 0 || neighborhood_size >= n_points) {
        // Original brute-force implementation
        float shift = compute_shift(q);
        float lambda = lambda_scale * maxExpArg / shift;
        //lambda = lambda_scale;
        
        float normalization = 0.0;
        vec3 grad_g_minus_lambda_n_g = vec3(0.0);
        float g = 0.0;
        vec3 n = vec3(0.0);
        
        vec3 center, axis;
        float R, r;
        
        for (int i = 0; i < n_points; i++) {
            vec3 p_i = fetch_point(i);
            vec3 diff = q - p_i;
            float dist = length(diff);
            
            float weight = exp(-lambda * (dist - shift));
            if (usePointAreas) {
                float area = fetch_texel(2*i, pointcloud).w;
                weight *= area;
            }
            
            get_torus_params(i, texture, center, axis, R, r);
            vec3 rVec = q - center;
            float v = length(cross(rVec, axis));
            
            vec2 u = vec2(v - R, dot(rVec, axis));
            float u_len = length(u);
            
            float d = u_len - abs(r);
            float s = sign(r);
            float g_z = s * d;
            
            float G_11 = (axis.x*axis.y*(center.y-q.y) + axis.x*axis.z*(center.z-q.z) 
                          - (axis.y*axis.y + axis.z*axis.z)*(center.x-q.x)) / v;
            float G_12 = (axis.x*axis.y*(center.x-q.x) + axis.y*axis.z*(center.z-q.z) 
                          - (axis.x*axis.x + axis.z*axis.z)*(center.y-q.y)) / v;
            float G_13 = (axis.x*axis.z*(center.x-q.x) + axis.y*axis.z*(center.y-q.y) 
                          - (axis.x*axis.x + axis.y*axis.y)*(center.z-q.z)) / v;
            
            vec2 w = u / u_len;
            vec3 grad_g = s * vec3(w.x*G_11 + w.y*axis.x, 
                                   w.x*G_12 + w.y*axis.y, 
                                   w.x*G_13 + w.y*axis.z);
            
            vec3 rHat = diff / dist;
            
            grad_g_minus_lambda_n_g += weight * (grad_g - lambda * rHat * g_z);
            g += weight * g_z;
            n += weight * rHat;
            normalization += weight;
        }
        
        vec3 A = grad_g_minus_lambda_n_g / normalization;
        float distance = g / normalization;
        vec3 B = lambda * n / normalization;

        gradient = A + distance * B;
        return distance;
    }
    
    RadiusStats stats;
    bvhGatherStats(q, neighborhood_radius, stats);
    
    float shift, lambda;
    
    TorusAccumulator acc;
    acc.g = 0.0;
    acc.grad_g_minus_lambda_n_g = vec3(0.0);
    acc.n = vec3(0.0);
    acc.normalization = 0.0;
    
    if (stats.count > 0) {
        // Compute shift and lambda from the points found in radius
        shift = 0.5 * stats.max_dist;
        lambda = lambda_scale * maxExpArg / shift;
        
        // lambda *= neighborhood_radius / sqrt(3.0);
        
        bvhAccumulateInRadius(q, neighborhood_radius, texture, lambda, shift, acc);
        
    } else {
        // Fallback to k-NN if no points in radius
        // First pass of k-NN to get max_dist, then accumulate
        float knn_max_dist;
        
        int indices[max_neighborhood_size];
        float dists[max_neighborhood_size];
        int k = min(neighborhood_size, max_neighborhood_size);
        int count = bvhKNearestNeighbors(q, k, max_neighborhood_size, indices, dists);
        
        if (count == 0) {
            gradient = vec3(0.0);
            return 0.0;
        }
        
        // Find max distance
        float max_dist = 0.0;
        for (int i = 0; i < count; i++) {
            max_dist = max(max_dist, dists[i]);
        }
        
        shift = 0.5 * max_dist;
        lambda = lambda_scale * maxExpArg / shift;
        
        // Accumulate
        for (int i = 0; i < count; i++) {
            accumulateTorusPoint(q, indices[i], texture, lambda, shift, acc);
        }
    }
    
    vec3 A = acc.grad_g_minus_lambda_n_g / acc.normalization;
    float distance = acc.g / acc.normalization;
    vec3 B = lambda * acc.n / acc.normalization;
    gradient = A + distance * B;
    
    return distance;
}

vec3 blendedPointColor(in vec3 q, in sampler2D texture, in float lambda_scale) {
    
    float shift = compute_shift(q);
    float lambda = lambda_scale * maxExpArg / shift;

    if (n_pc_bvh_nodes == 0 || neighborhood_size <= 0 || neighborhood_size >= n_points) {
        float normalization = 0.;
        vec3 color = vec3(0., 0., 0.);
        for ( int i = 0; i < n_points; i++ ) {
            vec3 p_i = fetch_point(i);
            float weight = exp(-lambda*(length(q-p_i)-shift));
            if (usePointAreas) {
                float area = fetch_texel(2*i, pointcloud).w;
                weight *= area;
            }
            color += weight * fetch_texel(i, texture).xyz;
            normalization += weight;
        }
        // Simply take a linear average.
        return color / normalization;
    }

    // BVH
    int indices[max_neighborhood_size];
    float dists[max_neighborhood_size];
    int count = bvhKNearestNeighbors(q, neighborhood_size, max_neighborhood_size, indices, dists);
    if (count == 0) return vec3(0.);

    float normalization = 0.;
    vec3 color = vec3(0., 0., 0.);
    for ( int idx = 0; idx < count; idx++ ) {
        int i = indices[idx];
        vec3 p_i = fetch_point(i);
        float weight = exp(-lambda*(length(q-p_i)-shift));
        if (usePointAreas) {
            float area = fetch_texel(2*i, pointcloud).w;
            weight *= area;
        }
        color += weight * fetch_texel(i, texture).xyz;
        normalization += weight;
    }
    return color / normalization;
}


// Compute the signed distance to the original triangle mesh.
float meshSDF(in sampler2D mesh, in sampler2D bvh_nodes, in sampler2D bvh_indices, in int n_bvh_nodes, vec3 q) {

    if (n_bvh_nodes == 0  || neighborhood_size <= 0 || neighborhood_size >= n_points) {
        float d = INFINITY;
        float gwn = 0.0;
        vec3 normal, cp;
        for (int i=0; i < n_faces; i++) {
            vec3 v0 = fetch_texel(3*i, shape).xyz;
            vec3 v1 = fetch_texel(3*i+1, shape).xyz;
            vec3 v2 = fetch_texel(3*i+2, shape).xyz;
            vec3 a = v0 - q;
            vec3 b = v1 - q;
            vec3 c = v2 - q;
            float a_norm = length(a);
            float b_norm = length(b);
            float c_norm = length(c);
            gwn += 2.0 * atan(dot(a,cross(b,c)), a_norm*b_norm*c_norm + dot(a,b)*c_norm + dot(b,c)*a_norm + dot(c,a)*b_norm);
            vec3 n;
            vec3 m = closestPointOnTriangle(q, v0, v1, v2, n);
            float dist = length(q-m);
            if (dist < d) {
                d = dist;
                normal = n;
                cp = m;
            }
        }
        // The threshold for GWN (inside/outside) depends on how the shape is oriented.
        float s = (gwn > 2.0*PI) ? -1. : 1.; // if surface is positively oriented
        //float s = (gwn < -2.0*PI) ? 1. : 1.; // if surface is negatively oriented
        //float s = sign(dot(normal, q-cp)); simple pseudonormal test, brittle
        return s * d;
    }

    // BVH-accelerated version
    // Get k-nearest triangles for pseudonormal sign computation
    int tri_indices[max_neighborhood_size];
    float tri_dists[max_neighborhood_size];
    int count = bvhKNearestTriangles(mesh, bvh_nodes, bvh_indices, n_bvh_nodes, q, neighborhood_size, max_neighborhood_size, tri_indices, tri_dists);
    
    if (count == 0) {
        // No triangles found
        return INFINITY;
    }
    
    // Closest triangle gives us unsigned distance
    float unsigned_dist = tri_dists[0];
    
    // Compute pseudonormal sign using k-nearest triangles
    float sign_sum = 0.0;
    for (int i = 0; i < count; i++) {
        int tri_idx = tri_indices[i];
        
        vec3 v0 = fetch_texel(3 * tri_idx, mesh).xyz;
        vec3 v1 = fetch_texel(3 * tri_idx + 1, mesh).xyz;
        vec3 v2 = fetch_texel(3 * tri_idx + 2, mesh).xyz;
        
        // Compute triangle normal
        vec3 e0 = v1 - v0;
        vec3 e1 = v2 - v0;
        vec3 tri_normal = normalize(cross(e0, e1));
        
        // Compute vector from triangle centroid to query
        vec3 c = (v0 + v1 + v2) / 3.0;
        vec3 to_query = q - c;
        
        // Pseudonormal test: if normal points toward query, we're outside
        sign_sum += sign(dot(tri_normal, to_query));
    }
    
    float avg_sign = sign_sum / float(count);
    float s = (avg_sign > 0.0) ? 1.0 : -1.0;
    
    return s * unsigned_dist;
}

// Compute the color of the i-th torus.
vec3 torusColor(int i) {
    float incr = (1.0+sqrt(5.0))/20.; // pick an increment that doesn't go into 1 evenly, so all tori are a different color 
    vec3 hsv = vec3(0., 0.5, 1.);
    hsv.x = mod(i*incr, 1.);
    return hsv2rgb(hsv);
}

bool intersectTorus(in sampler2D texture, in int idx, in vec3 ro, in vec3 rd, in float tmin, in float tmax, in int maxSteps, in float epsilon, out float t, out vec3 torusPosition, out vec3 torusNormal) {
    vec3 center, axis;
    float R, r;
    get_torus_params(idx, texture, center, axis, R, r);
    // sphere-trace
    t = tmin;
    for (int iters=0; iters < maxSteps && t < tmax; iters++) {
        vec3 pos = ro + t * rd;
        vec3 rVec = pos - center;
        vec2 u = vec2(length(cross(rVec, axis)) - R, dot(rVec, axis));
        float dist = abs(length(u) - abs(r)); // unsigned distance to torus
        if (dist < epsilon) {
            torusPosition = pos;
            vec3 centerline = center + R*normalize(rVec - dot(rVec, axis) * axis); // position along major circle
            torusNormal = pos - centerline;
            torusNormal /= length(torusNormal);
            return true;
        }
        t += dist;
     }
     return false;
}

/*
 * Return the index of the intersected torus.
 */
int intersecttori(in sampler2D texture, in vec3 ro, in vec3 rd, in float tmin, in float tmax, in int maxSteps, in float epsilon, out float t, out vec3 torusPosition, out vec3 torusNormal) {
    int idx = -1;
    float tMax = tmax;
    for (int i=0; i < n_points; i++) {
        if (intersectTorus(texture, i, ro, rd, tmin, tMax, maxSteps, epsilon, t, torusPosition, torusNormal)) {
            idx = i;
            tMax = t;
        }
    }
    return idx;
}


"""

# Shading and scene intersection stuff
COMMON_TEMPLATE = """
    const int stack_size = {stack_size};  // The maximum BVH depth allowed
    {COMMON_SHADER}

    in vec2 iResolution;
    uniform float screenWidth;
    uniform sampler2D learnedtori;
    uniform samplerCube cubemap;
    uniform float isobandFrequency, minDist, maxDist;

    uniform vec2 mouseCoord, cameraShift;
    uniform vec3 cutplaneDiag, cutplaneNormal, cameraPosition;
    uniform float zoom, lambdaScale, pointRadius, cutplaneDepth, isovalue, cutplaneRotationX, cutplaneRotationY;
    const int quantity = {quantity};
    uniform int cutplaneAxis;
    uniform bool freezeCutplane;

    /*
     * Approximate the gradient of true signed distance.
     * There are a few ways to do this: 
     *  - normal at closest point, or normalized vector to closest point (unstable)
     *  - smoothed normal at closest point via closest-point vector interpolation (more stable, albeit an approximation)
     *  - pixel-based finite differences
     * Going with the 2nd option.
     */ 
    vec3 meshGradient(in sampler2D mesh, in sampler2D bvh_nodes, in sampler2D bvh_indices, in int n_bvh_nodes, vec3 q) {{
        float shift = compute_shift(q);
        float lambda = lambdaScale * maxExpArg / shift;
        
        if (n_bvh_nodes == 0) {{
            // Fallback to brute force
            vec3 estimate = vec3(0.0, 0.0, 0.0);
            float normalization = 0.;
            for (int i=0; i < n_faces; i++) {{
                vec3 v0 = fetch_texel(3*i, mesh).xyz;
                vec3 v1 = fetch_texel(3*i+1, mesh).xyz;
                vec3 v2 = fetch_texel(3*i+2, mesh).xyz;
                vec3 e0 = v1 - v0;
                vec3 e1 = v2 - v0;
                vec3 n = 0.5*cross(e0, e1);
                // Approximate integral of ∫ exp(-λ|q-y|) dy with single-point quadrature. 
                vec3 c = (v0+v1+v2) / 3.0; // centroid
                float d = length(q - c);
                float area = length(n);
                float weight = exp(-lambda*(d-shift));
                estimate += n * weight; // normal is already area-weighted
                normalization += area * weight;
            }}
            return estimate/normalization;
        }}
        
        // BVH-accelerated version
        int tri_indices[max_neighborhood_size];
        float tri_dists[max_neighborhood_size];
        int count = bvhKNearestTriangles(mesh, bvh_nodes, bvh_indices, n_bvh_nodes, q, neighborhood_size, max_neighborhood_size, tri_indices, tri_dists);
        
        if (count == 0) {{
            return vec3(0.0);
        }}
        
        vec3 estimate = vec3(0.0, 0.0, 0.0);
        float normalization = 0.0;
        
        for (int i = 0; i < count; i++) {{
            int tri_idx = tri_indices[i];
            
            vec3 v0 = fetch_texel(3 * tri_idx, mesh).xyz;
            vec3 v1 = fetch_texel(3 * tri_idx + 1, mesh).xyz;
            vec3 v2 = fetch_texel(3 * tri_idx + 2, mesh).xyz;
            
            vec3 e0 = v1 - v0;
            vec3 e1 = v2 - v0;
            vec3 n = 0.5 * cross(e0, e1); // Area-weighted normal
            
            // Centroid for single-point quadrature
            vec3 c = (v0 + v1 + v2) / 3.0;
            float d = length(q - c);
            float area = length(n);
            float weight = exp(-lambda * (d - shift));
            
            estimate += n * weight;
            normalization += area * weight;
        }}
        
        return estimate / normalization;
    }}

    void activePhiAndGradient(in vec3 pos, out float phi, out vec3 gradient) {{

        if (quantity == 0) {{
            phi = meshSDF(shape, mesh_bvh_nodes, mesh_bvh_prim_indices, n_mesh_bvh_nodes, pos);
            gradient = meshGradient(shape, mesh_bvh_nodes, mesh_bvh_prim_indices, n_mesh_bvh_nodes, pos);
        }}
        else if (quantity == 1) {{
            phi = blendedTorusDistanceGradient(pos, learnedtori, lambdaScale, gradient);
        }}
        else if (quantity == 2) {{
            phi = selfNormalizedSignedLogSumExpGradient(pos, lambdaScale, gradient);
        }}
        else if (quantity == 3) {{
            phi = signedLogSumExpGradient(pos, lambdaScale, gradient);
        }}
        else if (quantity == 9) {{
            gradient = meshGradient(shape, mesh_bvh_nodes, mesh_bvh_prim_indices, n_mesh_bvh_nodes, pos);
            phi = 1.-length(gradient);
        }}
        else if (quantity == 6) {{
            phi = blendedTorusDistanceGradient(pos, learnedtori, lambdaScale, gradient);
            phi = 1.-length(gradient);
        }}
        else if (quantity == 7) {{
            phi = selfNormalizedSignedLogSumExpGradient(pos, lambdaScale, gradient);
            phi = 1.-length(gradient);
        }}
        else if (quantity == 8) {{
            phi = signedLogSumExpGradient(pos, lambdaScale, gradient);
            phi = 1.-length(gradient);
        }}
        else if (quantity == 4) {{
            windingNumber(pos, phi);
        }}
        else if (quantity == 5) {{
            phi = logSumExp(pos, lambdaScale);
        }}
    }}

    vec3 activeGradient(in vec3 pos) {{
        float phi;
        vec3 gradient;
        activePhiAndGradient(pos, phi, gradient);
        return gradient;
    }}

    float activePhi(in vec3 pos) {{
        float phi;
        vec3 gradient;
        activePhiAndGradient(pos, phi, gradient);
        return phi;
    }}

    vec2 toPixelCoordinates(in vec2 coord) {{
        //vec2 p = (coord - 0.5 * iResolution.xy) / (iResolution.y * zoom);
        vec2 p = (2.0*coord - iResolution.xy) / (iResolution.y * zoom);
        p -= cameraShift;
        return p;
    }}

    // code copied/adapted from https://www.shadertoy.com/view/4cfXWs
    vec3 diffuseShade( vec3 pos, vec3 normal, vec3 light, vec3 materialColor ) {{
        return materialColor * max( dot( normalize( light - pos ), normal ), 0.3 );
    }}

    vec3 fresnelShade( vec3 pos, vec3 ray, vec3 normal, vec3 materialColor ) {{
        //return vec3( pow( 1. - abs( dot( ray, normal ) ), 4. ) );
        //return vec3( pow( 1. - abs( dot( ray, normal ) ), 20. ) ); // less severe
        return vec3( pow( 1. - abs( dot( ray, normal ) ), 80. ) ); // less severe
    }}

    // (from https://www.shadertoy.com/view/4cfXWs)
    // Start of modified PBR code from https://www.shadertoy.com/view/XsfXWX
    // (taken from https://www.shadertoy.com/view/7tKfz1)

    float somestep(float t) {{
        return pow(t,4.0);
    }}

    vec3 textureAVG(samplerCube tex, vec3 tc) {{
        const float diff0 = 0.35;
        const float diff1 = 0.12;
        vec3 s0 = texture(tex,tc).xyz;
        vec3 s1 = texture(tex,tc+vec3(diff0)).xyz;
        vec3 s2 = texture(tex,tc+vec3(-diff0)).xyz;
        vec3 s3 = texture(tex,tc+vec3(-diff0,diff0,-diff0)).xyz;
        vec3 s4 = texture(tex,tc+vec3(diff0,-diff0,diff0)).xyz;
        
        vec3 s5 = texture(tex,tc+vec3(diff1)).xyz;
        vec3 s6 = texture(tex,tc+vec3(-diff1)).xyz;
        vec3 s7 = texture(tex,tc+vec3(-diff1,diff1,-diff1)).xyz;
        vec3 s8 = texture(tex,tc+vec3(diff1,-diff1,diff1)).xyz;
        
        return (s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8) * 0.111111111;
    }}

    vec3 textureBlurred(samplerCube tex, vec3 tc) {{
        vec3 r = textureAVG(tex,vec3(1.0,0.0,0.0));
        vec3 t = textureAVG(tex,vec3(0.0,1.0,0.0));
        vec3 f = textureAVG(tex,vec3(0.0,0.0,1.0));
        vec3 l = textureAVG(tex,vec3(-1.0,0.0,0.0));
        vec3 b = textureAVG(tex,vec3(0.0,-1.0,0.0));
        vec3 a = textureAVG(tex,vec3(0.0,0.0,-1.0));
            
        float kr = dot(tc,vec3(1.0,0.0,0.0)) * 0.5 + 0.5; 
        float kt = dot(tc,vec3(0.0,1.0,0.0)) * 0.5 + 0.5;
        float kf = dot(tc,vec3(0.0,0.0,1.0)) * 0.5 + 0.5;
        float kl = 1.0 - kr;
        float kb = 1.0 - kt;
        float ka = 1.0 - kf;
        
        kr = somestep(kr);
        kt = somestep(kt);
        kf = somestep(kf);
        kl = somestep(kl);
        kb = somestep(kb);
        ka = somestep(ka);    
        
        float d;
        vec3 ret;
        ret  = f * kf; d  = kf;
        ret += a * ka; d += ka;
        ret += l * kl; d += kl;
        ret += r * kr; d += kr;
        ret += t * kt; d += kt;
        ret += b * kb; d += kb;
        
        return ret / d;
    }}

    // GGX code from https://www.shadertoy.com/view/MlB3DV
    float G1V ( float dotNV, float k ) {{
        return 1.0 / (dotNV*(1.0 - k) + k);
    }}
    float GGX(vec3 N, vec3 V, vec3 L, float roughness, float F0) {{
        float alpha = roughness*roughness;
        vec3 H = normalize (V + L);

        float dotNL = clamp (dot (N, L), 0.0, 1.0);
        float dotNV = clamp (dot (N, V), 0.0, 1.0);
        float dotNH = clamp (dot (N, H), 0.0, 1.0);
        float dotLH = clamp (dot (L, H), 0.0, 1.0);

        float D, vis;
        float F;

        // NDF : GGX
        float alphaSqr = alpha*alpha;
        float pi = 3.1415926535;
        float denom = dotNH * dotNH *(alphaSqr - 1.0) + 1.0;
        D = alphaSqr / (pi * denom * denom);

        // Fresnel (Schlick)
        float dotLH5 = pow (1.0 - dotLH, 5.0);
        F = F0 + (1.0 - F0)*(dotLH5);

        // Visibility term (G) : Smith with Schlick's approximation
        float k = alpha / 2.0;
        vis = G1V (dotNL, k) * G1V (dotNV, k);

        return /*dotNL */ D * F * vis;
    }}

    // from https://www.shadertoy.com/view/7tKfz1
    vec3 normalShade(vec3 ray, vec3 normal, in vec3 specularLightPosition, vec3 materialColor) {{
        // material
        float metallic = 0.07;
        float roughness = 0.1;
        float fresnel_pow = mix(5.0, 3.5, metallic);
        vec3 color_mod = vec3(1.0);
        vec3 light_color = pow(texture(cubemap,vec3(1.0,0.0,0.0)).xyz * 1.2, vec3(2.2));

        // IBL
        vec3 ibl_diffuse = pow(textureBlurred(cubemap,normal), vec3(2.2));
        vec3 ibl_reflection = pow(textureBlurred(cubemap,reflect(ray,normal)), vec3(2.2));

        // fresnel
        float fresnel = max(1.0 - dot(normal,-ray), 0.0);
        fresnel = pow(fresnel,fresnel_pow);    

        // reflection        
        vec3 refl = pow(texture(cubemap,reflect(ray,normal)).xyz, vec3(2.2));
        refl = mix(refl,ibl_reflection,(1.0-fresnel)*roughness);
        refl = mix(refl,ibl_reflection,roughness);

        // specular
        vec3 light = normalize(specularLightPosition);
        float power = 1.0 / max(roughness * 0.4,0.01);
        vec3 spec = light_color * GGX(normal,-ray,light,roughness*0.7, 0.2);
        refl -= spec;

        // diffuse
        vec3 diff = ibl_diffuse * pow(materialColor, vec3(2.2));
        diff = mix(diff * color_mod,refl,fresnel);        

        vec3 color = min( mix(diff,refl * color_mod,metallic) + spec, vec3(1., 1., 1.) );
        return pow(color, vec3(1.0/2.2));
    }}

    vec3 shade(in vec3 pos, in vec3 ray, in vec3 normal, in vec3 light, in vec3 specularLightPosition, in vec3 materialColor) {{
        // return min( normalShade( ray, normal, materialColor ), vec3( 1 ) );
        return min(
                diffuseShade( pos, normal, light, materialColor )
                + 0.3 * normalShade( ray, normal, specularLightPosition, materialColor )
                + 0.6 * fresnelShade( pos, ray, normal, materialColor ),
                vec3(1., 1., 1.)
            );
    }}

    /* only intersect from positive direction */
    bool intersectPlane( vec3 ro, vec3 rd, vec3 n, float d, out float t ) {{
        // dot(n, ro + t rd - origin) == d
        t = ( d - dot( n, ro ) ) / dot( n, rd );
        return dot( rd, n ) < 0. && t >= 0.;
    }}

    bool intersectPointCloud( in vec3 ro, in vec3 rd, in float tmax, out float t, out vec3 p, out vec3 n, out int hit_idx) {{

        if (n_pc_bvh_nodes > 0) {{
            if (bvhIntersectPointCloud(ro, rd, 0.0, tmax, pointRadius, t, hit_idx, n)) {{
                p = ro + t * rd;
                return true;
            }}
            return false;
        }}

        t = tmax;
        float ti;
        vec3 ni;
        for ( int i = 0; i < n_points; i++ ) {{
            if ( intersectSphere( ro, rd, fetch_point(i), pointRadius, ti, ni ) ) {{
                if (ti < t) {{
                    t = ti; n = ni;
                }}
            }}
        }}
        if (t < tmax) {{
            p = ro + t * rd;
            return true;
        }}
    
        return false;
    }}

    bool intersectMesh(in sampler2D mesh, in sampler2D bvh_nodes, in sampler2D bvh_indices, in int n_bvh_nodes, in vec3 ro, in vec3 rd, in float tmin, in float tmax, in int maxSteps, in float epsilon, out float t, out vec3 pos, out vec3 normal) {{

        if (n_bvh_nodes > 0) {{
            int hit_tri_idx;
            vec2 hit_bary;
            
            if (bvhIntersectMesh(mesh, bvh_nodes, bvh_indices, n_bvh_nodes, ro, rd, tmin, tmax, t, hit_tri_idx, hit_bary)) {{
                // Compute hit position
                pos = ro + t * rd;
                
                // Compute geometric normal from triangle
                vec3 v0 = fetch_texel(3 * hit_tri_idx, mesh).xyz;
                vec3 v1 = fetch_texel(3 * hit_tri_idx + 1, mesh).xyz;
                vec3 v2 = fetch_texel(3 * hit_tri_idx + 2, mesh).xyz;
                
                vec3 edge1 = v1 - v0;
                vec3 edge2 = v2 - v0;
                normal = normalize(cross(edge1, edge2));
                
                return true;
            }}
            return false;
        }}

        t = tmin;
        for (int iters=0; iters < maxSteps && t < tmax; iters++) {{
            pos = ro + t * rd;
            float d = meshSDF(mesh, bvh_nodes, bvh_indices, n_bvh_nodes, pos);
            float d_abs = abs(d);
            if (d_abs < epsilon) {{
                normal = meshGradient(mesh, bvh_nodes, bvh_indices, n_bvh_nodes, pos);
                normal /= length(normal);
                return true;
            }}
            t += d_abs;
         }}
         return false;
    }}


    bool intersectShape(in vec3 ro, in vec3 rd, in float tmin, in float tmax, in int maxSteps, in float epsilon, out float t, out vec3 pos, out vec3 normal) {{

        return intersectMesh(shape, mesh_bvh_nodes, mesh_bvh_prim_indices, n_mesh_bvh_nodes, ro, rd, tmin, tmax, maxSteps, epsilon, t, pos, normal);
    }}

    bool intersectContourMesh(in vec3 ro, in vec3 rd, in float tmin, in float tmax, in int maxSteps, in float epsilon, out float t, out vec3 pos, out vec3 normal) {{
        return intersectMesh(isocontour, isomesh_bvh_nodes, isomesh_bvh_prim_indices, n_isomesh_bvh_nodes, ro, rd, tmin, tmax, maxSteps, epsilon, t, pos, normal);
    }}

    // code copied/adapted from https://www.shadertoy.com/view/wdffWj

    /* 
     * ro: position of the camera in world space 
     * ta: the point the camera is looking at in world space (target?)
     * cr: how much the camera is rotated (in radians) around its looking direction
     */
    mat3 setCamera( in vec3 ro, in vec3 ta, float cr ) {{
        vec3 cw = normalize(ta - ro); // looking direction
        vec3 cp = vec3(sin(cr), cos(cr), 0.0);
        vec3 cu = normalize(cross(cw, cp));
        vec3 cv = (cross(cu, cw));
        return mat3(cu, cv, cw);
    }}

    void toRayParameters(in vec2 coord, out vec3 ro, out vec3 rd) {{
        vec3 ta = vec3(0., 0., 0.); // camera target
        ro = cameraPosition; // camera position
        
        // camera-to-world transformation
        mat3 ca = setCamera( ro, ta, 0.0 );

        // pixel coordinates
        vec2 p = toPixelCoordinates(coord);

        // ray direction
        rd = ca * normalize(vec3(p, 4.5));
    }}

    // Helper: Build orthonormal basis aligned with global axes, given a primary axis
    void buildAlignedBasis(int axis, out vec3 u, out vec3 v, out vec3 n) {{
        if (axis == 0) {{
            // Normal along X
            n = vec3(1.0, 0.0, 0.0);
            u = vec3(0.0, 1.0, 0.0);  // Rotation axis 1 (Y-aligned)
            v = vec3(0.0, 0.0, 1.0);  // Rotation axis 2 (Z-aligned)
        }} else if (axis == 1) {{
            // Normal along Y
            n = vec3(0.0, 1.0, 0.0);
            u = vec3(1.0, 0.0, 0.0);  // Rotation axis 1 (X-aligned)
            v = vec3(0.0, 0.0, 1.0);  // Rotation axis 2 (Z-aligned)
        }} else {{
            // Normal along Z
            n = vec3(0.0, 0.0, 1.0);
            u = vec3(1.0, 0.0, 0.0);  // Rotation axis 1 (X-aligned)
            v = vec3(0.0, 1.0, 0.0);  // Rotation axis 2 (Y-aligned)
        }}
    }}

    // Helper: Rotate vector v by angle theta about axis
    vec3 rotateAboutAxis(vec3 v, vec3 axis, float theta) {{
        float c = cos(theta);
        float s = sin(theta);
        float t = 1.0 - c;
        
        // Rodrigues' rotation formula
        return v * c + cross(axis, v) * s + axis * dot(axis, v) * t;
    }}

    // Helper: get the rotated cut plane basis
    void getCutPlaneBasis(out vec3 normal, out vec3 u, out vec3 v, out vec3 center) {{

        // Get initial aligned basis
        vec3 u_init, v_init, n_init;
        buildAlignedBasis(cutplaneAxis, u_init, v_init, n_init);
        
        // Apply rotations: first about u (cutplaneRotationX), then about v (cutplaneRotationY)
        normal = rotateAboutAxis(n_init, u_init, cutplaneRotationX);
        u = rotateAboutAxis(u_init, u_init, cutplaneRotationX);
        v = rotateAboutAxis(v_init, u_init, cutplaneRotationX);
        
        // Second rotation about the rotated v axis
        normal = rotateAboutAxis(normal, v, cutplaneRotationY);
        u = rotateAboutAxis(u, v, cutplaneRotationY);
        // v stays the same after rotating about itself
        
        // Normalize to ensure orthonormality despite floating point errors
        normal = normalize(normal);
        u = normalize(u);
        v = normalize(v);

        float depth = freezeCutplane ? mix(1.0, -1.0, cutplaneDepth / screenWidth) : mix(1.0, -1.0, mouseCoord.x / screenWidth);
        center = n_init * depth;
    }}

    bool intersectCutPlane(in vec3 ro, in vec3 rd, in float epsilon, out float t) {{
        vec3 normal, u, v, center;
        getCutPlaneBasis(normal, u, v, center);
        
        float denom = dot(normal, rd);
        
        // Ray parallel to plane
        if (abs(denom) < epsilon) {{
            return false;
        }}
        
        // Intersect with infinite plane first
        t = (dot(normal, center) - dot(normal, ro)) / denom;
        
        if (t <= epsilon) {{
            return false;
        }}
        
        // Get intersection point
        vec3 hit_point = ro + t * rd;
        
        // Check if hit point is within rectangular bounds
        vec3 local_pos = hit_point - center;
        float u_coord = dot(local_pos, u);
        float v_coord = dot(local_pos, v);
        
        // Half-extents
        float half_width, half_height;
        if (cutplaneAxis == 0) {{
            // Normal along X, so use Y and Z dimensions
            half_width = cutplaneDiag.y;
            half_height = cutplaneDiag.z;
        }} else if (cutplaneAxis == 1) {{
            // Normal along Y, so use X and Z dimensions
            half_width = cutplaneDiag.x;
            half_height = cutplaneDiag.z;
        }} else {{
            // Normal along Z, so use X and Y dimensions
            half_width = cutplaneDiag.x;
            half_height = cutplaneDiag.y;
        }}
        
        // Check bounds
        if (abs(u_coord) > half_width || abs(v_coord) > half_height) {{
            return false;
        }}
        
        return true;
    }}

    /* Check if camera ray hits slice plane */
    

    /* https://iquilezles.org/articles/boxfunctions/ */
    bool intersectCutPlaneBox( in vec3 row, in vec3 rdw) {{
        float cuttingPlane = freezeCutplane ? mix(1.0, -1.0, cutplaneDepth / iResolution.x) : mix(1.0, -1.0, mouseCoord.x / iResolution.x);
        int z = cutplaneAxis;
        int x = (cutplaneAxis + 1) % 3;
        int y = (cutplaneAxis + 2) % 3;
        vec3 x_axis = vec3(0., 0., 0.); vec3 y_axis = vec3(0., 0., 0.); vec3 z_axis = cutplaneNormal;
        x_axis[x] = 1.0; y_axis[y] = 1.0;
        x_axis = normalize(x_axis - dot(x_axis, cutplaneNormal)*cutplaneNormal) * cutplaneDiag[x];
        y_axis = normalize(y_axis - dot(y_axis, cutplaneNormal)*cutplaneNormal) * cutplaneDiag[y];
        vec3 translate = cuttingPlane * cutplaneNormal;

        // txx is the world-to-box transformation
        // txi is the box-to-world transformation
        // rad is the half-length of the box
        mat4 txi = mat4(vec4(x_axis, 0.0), vec4(y_axis, 0.0), vec4(z_axis, 0.0), vec4(translate, 1.0));
        mat4 txx = inverse(txi);
        vec3 rad = vec3(cutplaneDiag[x], cutplaneDiag[y], min(cutplaneDiag[x], cutplaneDiag[y]));

        // convert from world to box space
        vec3 rd = (txx*vec4(rdw,0.0)).xyz;
        vec3 ro = (txx*vec4(row,1.0)).xyz;

        // ray-box intersection in box space
        vec3 m = 1.0/rd;
        vec3 s = vec3((rd.x<0.0)?1.0:-1.0,
                      (rd.y<0.0)?1.0:-1.0,
                      (rd.z<0.0)?1.0:-1.0);
        vec3 t1 = m*(-ro + s*rad);
        vec3 t2 = m*(-ro - s*rad);

        float tN = max( max( t1.x, t1.y ), t1.z );
        float tF = min( min( t2.x, t2.y ), t2.z );
        
        if( tN>tF || tF<0.0) return false;
        return true;
    }}

    bool intersectContour(in vec3 ro, in vec3 rd, in float tmin, in float tmax, in int maxSteps, in float epsilon, out float t, out vec3 pos, out vec3 normal) {{

        // check bounding sphere
        vec3 p = vec3(0.0,0.0,0.0);
        float diameter = length(cutplaneDiag);
        float radius = 0.8*diameter + isovalue;
        vec3 n;
        if (!intersectSphere(ro, rd, p, radius, t, n)) {{
            return false;
        }}

        t = tmin;
        for (int iters=0; iters < maxSteps && t < tmax; iters++) {{
            pos = ro + t * rd;
            float phi;
            vec3 gradient;
            
            activePhiAndGradient(pos, phi, gradient);

            phi -= isovalue;
            float phi_abs = abs(phi);

            if (abs(phi) < epsilon) {{
                normal = gradient/length(gradient);
                return true;
            }}
            t += phi_abs / length(gradient);
            //t += 0.9*phi_abs;
         }}
         return false;
    }}

    /* Compute the color to be displayed at the given point on the cut plane.*/
    float cutplanePhi(in vec3 pos) {{
        float phi;
        vec3 gradient;
        activePhiAndGradient(pos, phi, gradient);
        return phi;
    }}
"""


# A pass for computing quantities that depend on e.g. the mouse position, and stores them in a texture to be read by the main program.
INTERACTION_SHADER_TEMPLATE = """
    #version 330 core

    {COMMON_DYNAMIC}
    
    // Multiple output targets
    layout(location = 0) out vec4 output0;
    layout(location = 1) out vec4 output1;
    layout(location = 2) out vec4 output2;
    layout(location = 3) out vec4 output3;

    void main() {{

        vec3 ro, rd;
        toRayParameters(mouseCoord, ro, rd);
        float epsilon = 1e-2;
        float t;
        bool hitPlane = intersectCutPlane(ro, rd, epsilon, t);
        vec3 pos = ro + t*rd;


        if (hitPlane) {{
            float mouse_sdf_approx = activePhi(pos);
            vec3 mouse_grad_approx = activeGradient(pos);

            float mouse_sdf_true = meshSDF(shape, mesh_bvh_nodes, mesh_bvh_prim_indices, n_mesh_bvh_nodes, pos);
            vec3 mouse_grad_true = meshGradient(shape, mesh_bvh_nodes, mesh_bvh_prim_indices, n_mesh_bvh_nodes, pos);

            output0 += vec4(mouse_sdf_true, mouse_grad_true.x, mouse_grad_true.y, mouse_grad_true.x);
            output1 = vec4(mouse_sdf_approx, mouse_grad_approx.x, mouse_grad_approx.y, mouse_grad_approx.z);
            output2 = vec4(pos.x, pos.y, pos.z, 0.0);
        }}
    }}
"""

CUTPLANE_EXPORT_SHADER = """
#version 330 core

{COMMON_DYNAMIC}

in vec2 fragCoord;

layout(location = 0) out vec4 out_position;
layout(location = 1) out vec4 out_scalar;
layout(location = 2) out vec4 out_color;

uniform vec2 cutplane_resolution;

void main() {{
    
    out_position = vec4(0.);
    if (n_points < 0) {{
        out_position.x += minDist;
        out_position.x += maxDist;
        out_position.x += isobandFrequency;
    }}

    vec3 normal, u, v, center;
    getCutPlaneBasis(normal, u, v, center);
    
    // Determine dimensions based on cutplane axis
    float half_width, half_height;
    if (cutplaneAxis == 0) {{
        half_width = cutplaneDiag.y;
        half_height = cutplaneDiag.z;
    }} else if (cutplaneAxis == 1) {{
        half_width = cutplaneDiag.x;
        half_height = cutplaneDiag.z;
    }} else {{
        half_width = cutplaneDiag.x;
        half_height = cutplaneDiag.y;
    }}
    float width = 2.0 * half_width;
    float height = 2.0 * half_height;
    
    // Map UV to [-0.5, 0.5] range
    // gl_FragCoord.xy ranges from (0.5, 0.5) to (iResolution.x - 0.5, iResolution.y - 0.5)
    vec2 local_uv = (gl_FragCoord.xy - 0.5) / (cutplane_resolution - vec2(1.0, 1.0)) - 0.5;
    
    // Compute 3D position on cut plane
    vec3 pos = center + local_uv.x * width * u + local_uv.y * height * v;
    
    float phi = activePhi(pos);
    vec3 color = colormap_SDF(normalize_scalar(minDist, maxDist, phi)); 
    float dark_factor = 0.9;
    dark_factor *= 1.0 - 0.1*smoothstep(-0.01,0.01,sin(-isobandFrequency*phi ));
    color = darken_color(color, dark_factor);
    
    // Output
    out_position.xyz += pos;
    out_scalar.x = phi;
    out_color.xyz = color;
}}
"""

# Equivalent to ShaderToy's mainImage() function
# All braces are escaped (via double braces), because this string is used for string formatting in demo.py
MAIN_FRAGMENT_SHADER_TEMPLATE = """
    #version 150 core

    {COMMON_DYNAMIC}

    in vec2 fragCoord;
    out vec4 fragColor;
    uniform vec3 lightPosition, specularLightPosition;
    uniform float floorHeight;
    uniform bool vis_cutplane, vis_points, vis_normals;
    const bool vis_shape = {vis_shape}, vis_tori = {vis_tori}, vis_contour = {vis_contour};
    uniform bool vis_colors;

    /* Compute the color to be displayed at the given point on the cut plane. */
    vec3 cutplaneColor(in vec3 pos) {{

        vec3 color;
        
        float phi = activePhi(pos);

        // Render SDF values with isobands
        float dark_factor = 0.9;
        dark_factor *= 1.0 - 0.1*smoothstep(-0.01,0.01,sin(-isobandFrequency*phi ));
        vec3 col;
        if (quantity == 4) {{ // winding number
            col = colormap_SDF(normalize_scalar(minDist, maxDist, phi + 0.5)); // shift so that 0.5 is at 0
        }}
        else {{
            col = colormap_SDF(normalize_scalar(minDist, maxDist, phi)); 
        }} 
        color = darken_color(col, dark_factor);

        return color;
    }}

    /* Returns true if camera ray hits object in scene; also returns the color of the pixel. */
    bool intersectScene(in vec3 ro, in vec3 rd, in float tmin, in float tmax, out vec3 col) {{
        
        bool hit = false;
        vec3 pos, normal, vertexPos, vertexNormal;
        int maxSteps = 20;
        int hit_idx;
        float epsilon = 1e-2;
        float t;

        // point cloud
        if (vis_points && intersectPointCloud(ro, rd, tmax, t, vertexPos, vertexNormal, hit_idx)) {{
            vec3 baseColor = vec3( 0., 0., 0. );
            // Render point normals; color points with RGB
            if (vis_normals) {{
                baseColor = fetch_normal(hit_idx);
            }}
            else if (vis_colors) {{
                baseColor = fetch_texel(hit_idx, pointcolors).xyz;
            }}
            col = shade( vertexPos, rd, vertexNormal, lightPosition, specularLightPosition, baseColor );
            hit = true;
            tmax = t;
        }}

        // cut plane
        if (vis_cutplane && intersectCutPlane(ro, rd, epsilon, t) && t < tmax) {{
            pos = ro + t*rd;
            col = cutplaneColor(pos);
            hit = true;
            tmax = t;
        }}

        // original shape
        if (vis_shape && intersectShape(ro, rd, tmin, tmax, maxSteps, epsilon, t, pos, normal)) {{
            vec3 baseColor = (dot( normal, rd ) < 0.) ? (vec3(210, 180, 140) / 255.) : vec3(1.0, 0.0, 1.0);
            vec3 outwardNormal = dot( normal, rd ) > 0. ? -normal : normal;
            col = shade( pos, rd, outwardNormal, lightPosition, specularLightPosition, baseColor );
            col = sqrt(col);
            hit = true;
            tmax = t;
        }}

        // contoured surface
        if (vis_contour && intersectContour(ro, rd, tmin, tmax, 40, 1e-2, t, pos, normal)) {{
        //if (vis_contour && intersectContourMesh(ro, rd, tmin, tmax, maxSteps, epsilon, t, pos, normal)) {{
            //vec3 baseColor = (dot( normal, rd ) < 0.) ? vec3(0.266667, 0.439216, 0.568627) : vec3(0.992157, 0.431373, 0.337255); // nicole blue / red
            //vec3 baseColor = vec3(0.266667, 0.439216, 0.568627); // nicole blue
            vec3 baseColor = vec3(68.0/255.0, 74.0/255., 145.0/255.); // purple
            if (vis_colors) {{
                baseColor = blendedPointColor(pos, pointcolors, lambdaScale);
            }}
            vec3 outwardNormal = dot( normal, rd ) > 0. ? -normal : normal;
            col = shade( pos, rd, outwardNormal, lightPosition, specularLightPosition, baseColor );
            col = sqrt(col);
            hit = true;
            tmax = t;
        }}

        float torus_epsilon = 1e-4;
        if (vis_tori) {{
            vec3 torusPosition, torusNormal;
            int torusIdx = intersecttori(learnedtori, ro, rd, tmin, tmax, maxSteps, torus_epsilon, t, torusPosition, torusNormal);

            if (torusIdx > 0) {{
                vec3 baseColor = torusColor(torusIdx);
                vec3 outwardNormal = dot( torusNormal, rd ) > 0. ? -torusNormal : torusNormal;
                col = shade( torusPosition, rd, outwardNormal, lightPosition, specularLightPosition, baseColor );
                col = sqrt(col);
                hit = true;
                tmax = t;
            }}

            // If mouse is hovering over this torus, also highlight the corresponding point.
            float mouse_d;
            float mouse_tmax = 20.;
            vec3 mouse_ro, mouse_rd;
            toRayParameters(mouseCoord, mouse_ro, mouse_rd);
            vec3 hoveredTorusPosition, hoveredTorusNormal;
            int hoveredTorusIdx = intersecttori(learnedtori, mouse_ro, mouse_rd, 0., mouse_tmax, maxSteps, torus_epsilon, t, hoveredTorusPosition, hoveredTorusNormal);
            if (hoveredTorusIdx > 0) {{
                // Determine camera-ray intersection to hovered-over torus
                bool hit = intersectTorus(learnedtori, hoveredTorusIdx, ro, rd, tmin, tmax, maxSteps, epsilon, t, torusPosition, torusNormal);
                // Draw torus silhouette
                if (hit && dot(ro-torusPosition, torusNormal) < 2e-1*length(ro-torusPosition)) {{
                    col = vec3(0.0, 0.0, 0.0);
                }}
                // Highlight corresponding point
                float ti; vec3 ni;
                if (hit && intersectSphere( ro, rd, fetch_point(hoveredTorusIdx), 2.0*pointRadius, ti, ni )) {{
                    col = vec3(1.0, 0.0, 1.0);
                }}
            }}
        }}

        return hit;
    }}

    /* Stripped-down version for shadow rays. */
    bool intersectScene(in vec3 ro, in vec3 rd, in float tmin, in float tmax) {{

        vec3 pos, normal, vertexPos, vertexNormal;
        int maxSteps = 80;
        float epsilon = 1e-4;
        float t;
        if (vis_shape && intersectShape(ro, rd, tmin, tmax, maxSteps, epsilon, t, pos, normal)) {{
            return true;
        }}

        if (vis_contour && intersectContour(ro, rd, tmin, tmax, maxSteps, epsilon, t, pos, normal)) {{
        //if (vis_contour && intersectContourMesh(ro, rd, tmin, tmax, maxSteps, epsilon, t, pos, normal)) {{
            return true;
        }}
        return false;
    }}

    vec3 render(in vec3 ro, in vec3 rd, out float alpha) {{

        vec3 col = vec3(1.); // background color
        alpha = 0.; // transparent
        float tmin = 0.0;
        float tmax = 20.0;

        // ground shading
        vec3 groundNormal = vec3(0.0, 1.0, 0.0);
        float tGround;
        bool hitGround = intersectPlane( ro, rd, groundNormal, floorHeight, tGround );
        bool groundShadowed = false;
        if ( hitGround ) {{
            tmax = tGround;
            vec3 groundPos = ro + tGround * rd;
            vec3 lightDir = normalize( lightPosition - groundPos );
            groundShadowed = intersectScene(groundPos + 0.01 * lightDir, lightDir, tmin, length( lightPosition - groundPos ));
        }}
        
        // sphere-trace rest of the scene: point cloud, original surface, contoured surface, quantities on cutplane, etc. (if any)
        vec3 color;
        if (intersectScene(ro, rd, tmin, tmax, color)) {{
            alpha = 1.0;
            return color;
        }}
        if (groundShadowed) {{
            alpha = 1.0;
            return col * 0.8;
        }}
        return col;
    }}

    void main () {{


        vec3 ro, rd;
        float alpha;
        toRayParameters(fragCoord, ro, rd);
        vec3 color = render(ro, rd, alpha); // gamma
        fragColor += vec4(color, alpha);
    }}
"""
