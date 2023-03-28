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
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
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

vec2 getPoint(vec2 point, vec2 dir, float d) {
	vec2 newPoint;

	vec2 vec = dir * d;
	newPoint = point + vec;

	return newPoint;
};

float dis(vec2 p, vec2 q) {
	return sqrtf((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y));
}


class Circle {
	static const int nv = 100;
	vec2 center;
	vec2 coreCenter;
	float radius;
	float tmpradius;
	vec3 color;

	float sx, sy;		// scaling
	vec2 wTranslate=vec2(0,0);	// translation
	float phi=0;			// angle of rotation

public:
	Circle(vec2 cent, vec2 corCen, float r, vec3 col) {
		this->center = cent;
		this->coreCenter = corCen;
		this->radius = r;
		this->color = col;
		this->tmpradius = radius * (1 - dis(center + wTranslate, vec2(0, 0)));
	}

	vec2 getCenter() {
		return center;
	}

	float getRadius(){
		return radius;
	}

	void create() {
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)
		vec2 vertices[nv];
		for (int i = 0; i < nv; i++) {
			float fi = i * 2 * M_PI / nv;
			vertices[i] = vec2(cos(fi), sin(fi));
		}
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * nv,  // # bytes
			vertices,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed
	};

	mat4 Mrot() { // modeling transform
			// Translate to the origin
			mat4 Mtranslate1(1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				-coreCenter.x,-coreCenter.y, 0, 1);

			// Rotate
			mat4 Mrotate(cosf(phi), -sinf(phi), 0, 0,
				sinf(phi), cosf(phi), 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1); // rotation

			// Translate back to the original position
			mat4 Mtranslate2(1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				coreCenter.x, coreCenter.y, 0, 1);

			// Combine the transformations
			return Mtranslate1 * Mrotate * Mtranslate2;
	}

	mat4 Mmov() {
		mat4 Mtranslate(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			wTranslate.x, wTranslate.y, 0, 1); // translation

		return Mtranslate;
	}

	mat4 Msiz() {
		mat4 Mtranslate(1, 0, 0, 0,
			0, 1 , 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1); // translation
		//printf("%f\n", radius * dis(center, vec2(0, 0)));

		return Mtranslate;
	}

	void AddTranslation(vec2 wT, float szog) { 
		wTranslate = wTranslate + wT; 
		phi = phi + szog;
		tmpradius = radius * (1-dis(center + wTranslate , vec2(0, 0)));
		printf("tmprad: %f\n", tmpradius);
		printf("rad: %f\n", radius );
		printf("dis: %f\n",dis(center + wTranslate, vec2(0, 0)));
		printf("%f\n", radius * (1- dis(center + wTranslate, vec2(0, 0))));

	}

	void drawCircle() {
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, color.x, color.y, color.z); // 3 floats

		mat4 MVPtransf = { tmpradius, 0, 0, 0,    // MVP matrix, 
						  0, tmpradius, 0, 0,    // row-major!
							0, 0, 0, 0,
							 center.x, center.y, 0, 1 };

		mat4 MVPTransform = MVPtransf*Mrot()*Mmov()*Msiz();

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform);	// Load a 4x4 row-major float matrix to the specified location

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, nv /*# Elements*/);

		//// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		//mat4 MVPTransform = M();
		//gpuProgram.setUniform(MVPTransform, "MVP");
		//glBindVertexArray(vao);
		//glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, nv /*# Elements*/);
	};
};

class Hami{
public:
	std::vector<Circle> list;

	Hami(vec2 position, vec3 color) {
		//test
		list.push_back(Circle(position, position, 0.1f, color));
		
		//szemek
		list.push_back(Circle(	getPoint(list[0].getCenter(),		vec2(cos(1.0f), sin(1.0f)),			list[0].getRadius()), list[0].getCenter(),		 0.025f,		vec3(1.0f, 1.0f, 1.0f)));
		list.push_back(Circle(	getPoint(list[0].getCenter(),		vec2(-cos(1.0f), sin(1.0f)),		list[0].getRadius()), list[0].getCenter(),	 0.025f,		vec3(1.0f, 1.0f, 1.0f)));

		//pupilla
		list.push_back(Circle(getPoint(list[1].getCenter(), vec2(0,0), list[0].getRadius()), list[0].getCenter(), 0.01f, vec3(0.0f, 0.0f, 1.0f)));
		list.push_back(Circle(getPoint(list[2].getCenter(), vec2(0,0), list[0].getRadius()), list[0].getCenter(), 0.01f, vec3(0.0f, 0.0f, 1.0f)));

		//szaj
		list.push_back(Circle(getPoint(list[0].getCenter(), vec2(0, 0.88f), list[0].getRadius()), list[0].getCenter(),0.03f, vec3(0.0f, 0.0f, 0.0f)));

	}


	void createHami() {
		for (int i = 0; i < list.size(); i++) {
			list[i].create();
		}
	}

	void drawHami() {
		for (int i = 0; i < list.size(); i++) {
			list[i].drawCircle();
		}
	}

	void moveHami(vec2 dir, float sec) {
		for (int i = 0; i < list.size(); i++) {
			list[i].AddTranslation(dir, sec);
		}
	}
};

//Circle circle(vec2(0.5f, 0.5f), 0.5f);
Hami hamip(vec2(-0.5f, -0.3f), vec3(1.0f, 0.0f, 0.0f));
Hami hamiz(vec2(0.5f, 0.3f), vec3(0.0f, 1.0f, 0.0f));
Circle palya = Circle(vec2(0,0), vec2(0, 0), 1, vec3(0.0f, 0.0f, 0.0f));
Circle kp = Circle(vec2(0.3f, 0.3f), vec2(0, 0), 0.1f, vec3(1.0f, 0.0f, 0.0f));


// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	palya.create();
	//kp.create();

	hamip.createHami();
	hamiz.createHami();


	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.5f, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer
	
	palya.drawCircle();
	//kp.drawCircle();

	//printf("%f\n", dis(kp.getCenter(), vec2(0, 0)));

	hamip.drawHami();
	hamiz.drawHami();


	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
	/*case 'a': hamip.moveHami(vec2(-0.01f, 0), 0); printf("Pressed a\n"); break;
	case 'w': hamip.moveHami(vec2(0, 0.01f), 0); printf("Pressed w\n"); break;
	case 's': hamip.moveHami(vec2(0, -0.01f), 0); printf("Pressed s\n"); break;
	case 'd': hamip.moveHami(vec2(0.01f,0), 0); printf("Pressed d\n"); break;
	case 'e': hamip.moveHami(vec2(0, 0), 0.5); printf("Pressed e\n"); break;
	case 'q': hamip.moveHami(vec2(0, 0), -0.5); printf("Pressed q\n"); break;*/

	case 'a': hamip.moveHami(vec2(-0.01f, 0), 0); break;
	case 'w': hamip.moveHami(vec2(0, 0.01f), 0); break;
	case 's': hamip.moveHami(vec2(0, -0.01f), 0); break;
	case 'd': hamip.moveHami(vec2(0.01f, 0), 0); break;
	case 'e': hamip.moveHami(vec2(0, 0), 0.5); break;
	case 'q': hamip.moveHami(vec2(0, 0), -0.5); break;
	//case 't': printf("%f\n", dis(kp.getCenter(), vec2(0, 0))); break;
	case 't': printf("%f\n", kp.getCenter().x); break;

	}
	glutPostRedisplay();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	//printf("%f\n", sec);
	//kp.AddTranslation(vec2(0.001f,0), 0);	// animate the triangle object
	//hamip.moveHami(vec2(+0.0001, 0));
	//hamip.moveHami(vec2(0, +0.0001));

	glutPostRedisplay();
	
}