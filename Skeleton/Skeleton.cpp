//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Nagy Zalán
// Neptun : V9T3UL
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	
	vec2 toPoincare(vec2 point) {
	float r = length(point);
	return 2.0f * point / (1.0f + r * r);
	}
	
	void main() {
		//vec2 p = toPoincare(vp);
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel


	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";


GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao, vbo;// virtual world on the GPU
const int nv = 360;

vec2 pvertices[100];
void palyaCreate() {
	glGenVertexArrays(1, &vao);  // get 1 vao id
	glBindVertexArray(vao);      // make it active

	glGenBuffers(1, &vbo);    // Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);


	for (int i = 0; i < 100; i++) {
		float fi = i * 2 * M_PI / 100;
		pvertices[i] = vec2(cos(fi), sin(fi));
	}

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL);               // stride, offset: tightly packed
}
void palyaDraw() {
	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 0, 0, 0); // 3 floats


	glBindVertexArray(vao);

	glBufferData(GL_ARRAY_BUFFER,  // Copy to GPU target
		sizeof(vec2) * 100,  // # bytes
		pvertices,           // address
		GL_STATIC_DRAW);    // we do not change later

	glUseProgram(gpuProgram.getId());
	glDrawArrays(GL_TRIANGLE_FAN, 0, nv); // Draw call
}

float hdot(vec3 v, vec3 u) {
	return u.x * v.x + u.y * v.y - u.z * v.z;
}

float hlength(const vec3& v) { 
	return sqrtf(hdot(v, v));
}

vec3 hnormalize(const vec3& v) { 
	return v * (1 / hlength(v));
}

vec3 getjovec(vec3 p) { //egy olyan vektort ad vissza, amire v*p=0;
	vec3 v;
	v.x = p.x;
	v.y = p.y;
	v.z= (p.x * p.x + p.y * p.y) / p.z;
	v = hnormalize(v);
	return v;
}

vec3 hcross(vec3 p, vec3 q) {
	vec3 v1 = vec3(p.x, p.y, -1 * p.z);
	vec3 v2 = vec3(q.x, q.y, -1 * q.z);
	vec3 v = cross(v1, v2);
	v = hnormalize(v);
	return v;
}

vec2 toEukl(vec3 point) {
	vec2 vek = vec2(point.x / (point.z + 1), point.y / (point.z + 1));
	return vek;
}

void korrigal(vec3& p) {
	p.z = sqrtf(p.x * p.x + p.y * p.y + 1);
}

//1.fuggveny
vec3 meroleges(vec3 p) {
	vec3 q = getjovec(p);
	return hcross(p, q);
}

//2.fuggveny
void mozgatasHelyre(vec3 p, vec3& ujp, vec3 s, vec3& ujs, float t) {
	ujp = p * coshf(t) + hnormalize(s) * sinhf(t);
	korrigal(ujp);
	ujs= hnormalize(p * sinhf(t) + s * coshf(t));
}

//3.fuggveny
void pontIranyaTavolsaga(vec3 p, vec3 q, vec3& irany, float& tav) {
	irany = hnormalize((q - p * coshf(0.5)) / sinhf(0.5));
	tav = hlength(p-q);
}

//4.fuggveny
void pontEloallitasa(vec3 p, vec3& ujp, vec3 s, float t) {
	ujp = p * coshf(t) + hnormalize(s) * sinhf(t);
	korrigal(ujp);
}

//5.fuggveny
vec3 elforgat(vec3 q, vec3 meroleges, float rad) {
	return hnormalize((q * cosf(rad) + meroleges * sinf(rad)));
}


void korbeforgat(vec3 *p, vec3 center, float radius) { //körbeforgat egy vektrot egy pont körül
	vec3 q = getjovec(center);
	vec3 m = meroleges(center);
	vec3 elf;

	for (int i = 0; i < nv; i++) {
		elf = elforgat(q, m, 2 * M_PI / nv);
		p[i] = center * cosh(radius) + hnormalize(elf) * sinh(radius);
		q = elf;
		m = hcross(center, q);
	}
}

class LineStrip {
	std::vector<vec2> vpoints;

public:
	void create() {
		glGenVertexArrays(1, &vao);  // get 1 vao id
		glBindVertexArray(vao);      // make it active

		glGenBuffers(1, &vbo);    // Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL);               // stride, offset: tightly packed
	}

	void AddPoint(vec3 c) {
		vpoints.push_back(toEukl(c));
	}

	void Draw() {
		if (vpoints.size() > 0) {
			int location = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform3f(location, 1, 1, 1); // 3 floats


			glBindVertexArray(vao);

			glBufferData(GL_ARRAY_BUFFER,  // Copy to GPU target
				sizeof(vec2) * vpoints.size(),  // # bytes
				&vpoints[0],           // address
				GL_STATIC_DRAW);    // we do not change later

			glUseProgram(gpuProgram.getId());
			glDrawArrays(GL_LINE_STRIP, 0, vpoints.size()); // Draw call
		}
	}
};

class Circle {
	float radius;
	vec3 center;
	vec3 color;
	vec3 irany;
	float rad=0;
	vec2 vertices[nv];
	vec3 verticeshy[nv];
public:
	void setCenter(float x, float y) {
		center.x = x;
		center.y = y;
		center.z = sqrtf(x * x + y * y + 1);
		irany = getjovec(center);
	}

	void setRadius(float r) {
		radius = r;
	}

	void setColor(vec3 c) {
		color = c;
	}

	vec3 getVerticeshy(int index) {
		return verticeshy[index];
	}

	vec3 getirany() {
		return irany;
	}

	vec3 getCenter() {
		return center;
	}

	float getRadius() {
		return radius;
	}

	void vetites() {
		for (int i = 0; i < nv; i++) {
			vertices[i] = toEukl(verticeshy[i]);
		}
	}

	void mozgas(float d){
		mozgatasHelyre(center, center, hnormalize(irany), irany, d);
		
	}

	void forgas(float r) {
		irany = elforgat(irany, hcross(center, irany), r);
	}

	void korbemegy() {
		float d = 0.001;
		float r = 0.0025;
		mozgas(d);
		forgas(r);
	}
	
	void create() {
		glGenVertexArrays(1, &vao);  // get 1 vao id
		glBindVertexArray(vao);      // make it active

		glGenBuffers(1, &vbo);    // Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		korbeforgat(verticeshy, center, radius);


		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL);               // stride, offset: tightly packed
	}

	void draw() {


		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, color.x, color.y, color.z); // 3 floats

		mat4 MVPtransf = { 1, 0, 0, 0,    // MVP matrix, 
						  0, 1, 0, 0,    // row-major!
						  0, 0, 1, 0,
						   0, 0, 0, 1 };

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, MVPtransf);	// Load a 4x4 row-major float matrix to the specified location

		korbeforgat(verticeshy, center, radius);
		vetites();
		
		glBindVertexArray(vao);

		glBufferData(GL_ARRAY_BUFFER,  // Copy to GPU target
			sizeof(vec2) * nv,  // # bytes
			vertices,           // address
			GL_STATIC_DRAW);    // we do not change later

		glUseProgram(gpuProgram.getId());
		glDrawArrays(GL_TRIANGLE_FAN, 0, nv); // Draw call
		
	}
};

class Hami {
	LineStrip nyal;
	Circle test;
	Circle szaj;
	Circle szem1;
	Circle szem2;
	Circle pupilla1;
	Circle pupilla2;
	vec3 color;
	vec3 center;
	vec3 hovanez;// = vec3(-0.6, -0.8, sqrtf(0.6 * 0.6 + 0.8 * 0.8 + 1));
	float szajmeret = 0;
	bool no = true;
public:
	Hami(vec3 c, vec2 kp) {
		color = c;
		center.x = kp.x;
		center.y = kp.y;
		center.z = sqrtf(center.x * center.x + center.y * center.y + 1);
	}

	vec3 getszajcenter() {
		return center;
	}

	void sethovanez(vec3 v) {
		hovanez = v;
	}

	void create() {
		nyal.create();

		test.setColor(color);
		test.setRadius(0.2f);
		test.setCenter(center.x, center.y);
		test.create();

		szaj.setColor(vec3(0, 0, 0));
		szaj.create();
		
		szem1.setRadius(0.05f);
		szem1.setColor(vec3(1, 1, 1));
		szem1.create();

		szem2.setRadius(0.05f);
		szem2.setColor(vec3(1, 1, 1));
		szem2.create();

		pupilla1.setRadius(0.02f);
		pupilla1.setColor(vec3(0, 0, 1));
		pupilla1.create();

		pupilla2.setRadius(0.02f);
		pupilla2.setColor(vec3(0, 0, 1));
		pupilla2.create();


	}
	void draw() {
		vec3 v; //segedvaltozo
		vec3 u; //segedvaltozo
		float d; //segedvaltozo
		nyal.Draw();
		test.draw();


		u = elforgat(test.getirany(), hcross(test.getCenter(), test.getirany()), -M_PI / 4);
		pontEloallitasa(test.getCenter(), v, hnormalize(u), test.getRadius());
		szem1.setCenter(v.x, v.y);
		szem1.draw();

		u = elforgat(test.getirany(), hcross(test.getCenter(), test.getirany()), M_PI / 4);
		pontEloallitasa(test.getCenter(), v, hnormalize(u), test.getRadius());
		szem2.setCenter(v.x, v.y);
		szem2.draw();


		pontIranyaTavolsaga(szem1.getCenter(), hovanez, u, d);
		pontEloallitasa(szem1.getCenter(), v, hnormalize(u), szem1.getRadius());
		pupilla1.setCenter(v.x, v.y);
		pupilla1.draw();

		pontIranyaTavolsaga(szem2.getCenter(), hovanez, u, d);
		pontEloallitasa(szem2.getCenter(), v, hnormalize(u), szem2.getRadius());
		pupilla2.setCenter(v.x, v.y);
		pupilla2.draw();


		pontEloallitasa(test.getCenter(), v, hnormalize(test.getirany()), test.getRadius());
		szaj.setCenter(v.x, v.y);
		center = v;
		szaj.setRadius(szajmeret);
		no ? szajmeret += 0.00008 : szajmeret-=0.00008;
		if (szajmeret > 0.08) { no = false; }
		if (szajmeret < 0.001) { no = true; }
		szaj.draw();
	}

	void mozgas(float d) {
		test.mozgas(d);
		nyal.AddPoint(test.getCenter());
	}

	void forgas(float d) {
		test.forgas(d);
	}

	void korbemegy() {
		float d = 0.001;
		float r = 0.0025;
		mozgas(d);
		forgas(r);
	}

};

Hami zold = Hami(vec3(0, 1, 0), vec2(0.6, 0.3));
Hami piros = Hami(vec3(1, 0, 0), vec2(-0.6, 0.8));

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	palyaCreate();
	
	piros.create();
	zold.create();
	
	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.51f, 0.51f, 0.51f, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer

	palyaDraw();

	piros.sethovanez(zold.getszajcenter());
	piros.draw();

	zold.sethovanez(piros.getszajcenter());
	zold.draw();
	
	glutSwapBuffers(); // exchange buffers for double buffering
}


// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
	case 'e':
			piros.mozgas(0.1);
		break;
	case 's':
		piros.forgas(M_PI/2);
		break;
	case 'f':
		piros.forgas(-0.1);
		break;
	}
	glutPostRedisplay();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
}

// Idle event indicating that some time elapsed: do animation here

void onIdle() {
	zold.korbemegy();
	glutPostRedisplay();
}