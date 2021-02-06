#version 430 core

in layout(location=0) vec3 position;
in layout(location=1) vec4 color;
in layout(location=2) vec3 normal;
uniform layout(location=3) mat4 mat_t;
uniform layout(location=4) mat4 model_mat;

out layout(location=1) vec4 out_color;
out layout(location=2) vec3 out_normal;

void main()
{
    vec4 new_position = mat_t * vec4(position,1.0);
    out_normal = normalize(mat3(model_mat) * normal);
    out_color = color;
    gl_Position = new_position;
}