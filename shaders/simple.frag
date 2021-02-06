#version 430 core

// in vec4 ourColor;
in layout(location=2) vec3 in_normal;
in layout(location=1) vec4 in_color;

out vec4 color;

void main()
{
    vec3 lightDirection = normalize(vec3(0.8,-0.5,0.6));
    float c = max(0, dot(in_normal, -lightDirection));
    color = vec4(in_color.rgb * c, 1.0);
}