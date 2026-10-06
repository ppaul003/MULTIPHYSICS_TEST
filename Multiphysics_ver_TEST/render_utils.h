#ifndef RENDERER_UTILS_H
#define RENDERER_UTILS_H

#define STRINGIFY(A) #A

// -----------------------------------------------------------------------------
// PARTICLE SPHERE SHADERS
// -----------------------------------------------------------------------------
// Header-only version.
//
// IMPORTANT:
// These are marked static so this header can be included by more than one
// translation unit without causing duplicate-symbol linker errors.
// -----------------------------------------------------------------------------

static const char* vertexShader = STRINGIFY(
	attribute float radius;
	uniform float pointScale;
	uniform float densityScale;
	uniform float densityOffset;

	void main() {
		vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
		float dist = length(posEye);
		gl_PointSize = radius * (pointScale / dist);

		gl_TexCoord[0] = gl_MultiTexCoord0;
		gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

		gl_FrontColor = gl_Color;
	}
);

static const char* spherePixelShader = STRINGIFY(
	void main() {
		const vec3 lightDir = vec3(0.577, 0.577, 0.577);

		vec3 N;
		N.xy = gl_TexCoord[0].xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
		float mag = dot(N.xy, N.xy);

		if (mag > 1.0) discard;
		N.z = sqrt(1.0 - mag);

		float diffuse = max(0.0, dot(lightDir, N));

		gl_FragColor = gl_Color * diffuse;
	}
);

// -----------------------------------------------------------------------------
// LIGHT / EMISSIVE PARTICLE SHADERS
// -----------------------------------------------------------------------------

static const char* lightVertShader = STRINGIFY(
	attribute float radius;
	uniform float pointScale;

	void main() {
		vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
		float dist = length(posEye);

		gl_PointSize = radius * (pointScale / dist);

		gl_TexCoord[0] = gl_MultiTexCoord0;
		gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

		gl_FrontColor = gl_Color;
	}
);

static const char* lightFragShader = STRINGIFY(
	uniform float emissiveIntensity;
	void main() {
		vec2 uv = gl_TexCoord[0].xy * 2.0 - 1.0;
		float r2 = dot(uv, uv);

		if (r2 > 1.0)
			discard;

		float core = sqrt(1.0 - r2);

		float glow = 1.0 - r2;
		glow = glow * glow;

		float intensity = core + glow * 1.5;

		vec4 color = gl_Color;

		vec3 emissive = color.rgb * intensity * emissiveIntensity;
		float alpha = color.a * clamp(glow + core * 0.5, 0.0, 1.0);

		gl_FragColor = vec4(emissive, alpha);
	}
);
// -----------------------------------------------------------------------------
// PARTICLE MESH / OBJ SHADERS
// -----------------------------------------------------------------------------
// Directional Lambert shading for triangle meshes.
//
// Attribute layout intended for the later mesh-VBO checkpoint:
//
//     location 0 -> position
//     location 1 -> normal
//     queried      -> texcoord
//
// Lighting is evaluated in eye space because gl_NormalMatrix transforms
// object-space normals into the current model-view coordinate system.
// -----------------------------------------------------------------------------

static const char* modelVertexShader = STRINGIFY(
	attribute vec3 position;
	attribute vec3 normal;
	attribute vec2 texcoord;

	varying vec3 vNormalEye;
	varying vec2 vTexcoord;

	void main() {
		vNormalEye = normalize(gl_NormalMatrix * normal);
		vTexcoord = texcoord;

		gl_Position = gl_ModelViewProjectionMatrix *
			vec4(position, 1.0);
	}
);

static const char* modelFragmentShader = STRINGIFY(
	varying vec3 vNormalEye;
	varying vec2 vTexcoord;
	
	uniform vec4 uColor;
	uniform vec3 uLightDir;
	uniform float uAmbient;
	uniform sampler2D uBaseColorTexture;
	uniform int uUseBaseColorTexture;
	uniform int uAlphaMask;
	uniform float uAlphaCutoff;
	uniform sampler2D uEmissiveTexture;
	uniform int uUseEmissiveTexture;
	uniform vec3 uEmissiveFactor;
	uniform float uEmissiveIntensity;

	void main() {
		vec3 N = normalize(vNormalEye);
		vec3 L = normalize(uLightDir);

		float diffuse = max(dot(N, L), 0.0);
		float lighting = clamp(uAmbient + (1.0 - uAmbient) * diffuse, 0.0, 1.0);
		vec4 sampledBaseColor = uUseBaseColorTexture != 0
			? texture2D(uBaseColorTexture, vTexcoord)
			: vec4(1.0);
		vec4 surfaceColor = sampledBaseColor * uColor;
		if (uAlphaMask != 0 && surfaceColor.a < uAlphaCutoff) discard;

		vec3 emissiveSample = uUseEmissiveTexture != 0
			? texture2D(uEmissiveTexture, vTexcoord).rgb
			: vec3(1.0);
		vec3 emissive = emissiveSample * uEmissiveFactor * uEmissiveIntensity;
		gl_FragColor = vec4(surfaceColor.rgb * lighting + emissive, surfaceColor.a);

	}

);
// -----------------------------------------------------------------------------
// RENDERER UTILITY HELPERS
// -----------------------------------------------------------------------------

namespace RenderUtils {

	inline int clampi(int v, int lo, int hi) {
		return (v < lo) ? lo : (v > hi ? hi : v);
	}

	inline bool is_major(int i, int majorEvery) {
		return (majorEvery > 0) ? ((i % majorEvery) == 0) : true;
	}

	inline void draw_aabb_wire(const glm::vec3& mn, const glm::vec3& mx) {
		glBegin(GL_LINES);

		// bottom
		glVertex3f(mn.x, mn.y, mn.z); glVertex3f(mx.x, mn.y, mn.z);
		glVertex3f(mx.x, mn.y, mn.z); glVertex3f(mx.x, mn.y, mx.z);
		glVertex3f(mx.x, mn.y, mx.z); glVertex3f(mn.x, mn.y, mx.z);
		glVertex3f(mn.x, mn.y, mx.z); glVertex3f(mn.x, mn.y, mn.z);

		// top
		glVertex3f(mn.x, mx.y, mn.z); glVertex3f(mx.x, mx.y, mn.z);
		glVertex3f(mx.x, mx.y, mn.z); glVertex3f(mx.x, mx.y, mx.z);
		glVertex3f(mx.x, mx.y, mx.z); glVertex3f(mn.x, mx.y, mx.z);
		glVertex3f(mn.x, mx.y, mx.z); glVertex3f(mn.x, mx.y, mn.z);

		// verticals
		glVertex3f(mn.x, mn.y, mn.z); glVertex3f(mn.x, mx.y, mn.z);
		glVertex3f(mx.x, mn.y, mn.z); glVertex3f(mx.x, mx.y, mn.z);
		glVertex3f(mx.x, mn.y, mx.z); glVertex3f(mx.x, mx.y, mx.z);
		glVertex3f(mn.x, mn.y, mx.z); glVertex3f(mn.x, mx.y, mx.z);

		glEnd();
	}

	inline void draw_rect_wire_xy(float x0, float y0, float x1, float y1, float z) {
		glBegin(GL_LINES);
		glVertex3f(x0, y0, z); glVertex3f(x1, y0, z);
		glVertex3f(x1, y0, z); glVertex3f(x1, y1, z);
		glVertex3f(x1, y1, z); glVertex3f(x0, y1, z);
		glVertex3f(x0, y1, z); glVertex3f(x0, y0, z);
		glEnd();
	}

	inline void draw_rect_wire_xz(float x0, float z0, float x1, float z1, float y) {
		glBegin(GL_LINES);
		glVertex3f(x0, y, z0); glVertex3f(x1, y, z0);
		glVertex3f(x1, y, z0); glVertex3f(x1, y, z1);
		glVertex3f(x1, y, z1); glVertex3f(x0, y, z1);
		glVertex3f(x0, y, z1); glVertex3f(x0, y, z0);
		glEnd();
	}

	inline void draw_rect_wire_yz(float y0, float z0, float y1, float z1, float x) {
		glBegin(GL_LINES);
		glVertex3f(x, y0, z0); glVertex3f(x, y1, z0);
		glVertex3f(x, y1, z0); glVertex3f(x, y1, z1);
		glVertex3f(x, y1, z1); glVertex3f(x, y0, z1);
		glVertex3f(x, y0, z1); glVertex3f(x, y0, z0);
		glEnd();
	}
}
#endif
