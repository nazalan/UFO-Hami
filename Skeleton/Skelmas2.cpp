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
// Nev    : Fenyvesi Péter
// Neptun : E4P6FN
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

	void main() {
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
unsigned int vao;	   // virtual world on the GPU

const int circleDefinition = 100;
const int arcDefinition = 300;

const float circleRed = 0.19f;
const float circleGreen = 0.66f;
const float circleBlue = 0.33f;

const float arcRed = 0.8f;
const float arcGreen = 0.1f;
const float arcBlue = 0.1f;

const float fillRed = 0.78f;
const float fillGreen = 0.46f;
const float fillBlue = 0.18f;

int clickCounter = 0;
int arcCounter = 0;
int printCounter = 0;

vec2 p1, p2, p3;
vec2 c1, c2, c3;

float aAngle, bAngle, cAngle;
const float radConv = 57.2957795f; // ==> 180 / M_PI;

float calcDist(vec2 p, vec2 q) {
	return sqrtf((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y));
}

float calculateSirius(vec2 p, vec2 q) {
	float dist = calcDist(p, q);
	float len = fabs((q.x * q.x) + (q.y * q.y));
	return dist / (1 - ((len) * (len)));
}

float calculateLengthOfArc(vec2 Arr[], int arrSize) {
	float summa = 0.0f;
	for (int i = 0; i < arrSize - 1; i++) {
		summa += calculateSirius(Arr[i], Arr[i + 1]);
	}
	return summa;
}

float calculateInnerAngle(vec2 center1, vec2 center2, vec2 commonPoint) {
	vec2 aVec = commonPoint - center1;
	vec2 bVec = commonPoint - center2;
	float aVec_len = sqrtf((aVec.x * aVec.x) + (aVec.y * aVec.y));
	float bVec_len = sqrtf((bVec.x * bVec.x) + (bVec.y * bVec.y));
	float results = fabs(M_PI - acosf(((aVec.x * bVec.x) + (aVec.y * bVec.y)) / (aVec_len * bVec_len)));
	if (results > 0.8 * M_PI) {
		results = fabs(results - M_PI);
	}
	return results;
}

vec2 calcCenter(vec2 pont1, vec2 pont2) {
	vec2 p = pont1;
	vec2 q = pont2;

	float x = (p.x * p.x + p.y * p.y - p.y / q.y * q.x * q.x - p.y / q.y - p.y * q.y + 1) / (2 * (p.x - p.y * q.x / q.y));
	float y = (q.x * q.x + q.y * q.y - q.x / p.x * p.y * p.y - q.x / p.x - p.x * q.x + 1) / (2 * (q.y - q.x * p.y / p.x));

	return vec2(x, y);
}

void swap(vec2 a, vec2 b) {
	vec2 temp = a;
	a = b;
	b = temp;
}

bool checkIfUnique(std::vector<vec2>& array, vec2 isIt) {
	for (int i = 0; i < array.size(); i++) {
		float x_diff = fabs(array[i].x - isIt.x);
		float y_diff = fabs(array[i].y - isIt.y);
		if ((x_diff <= 0.0001f) && (y_diff <= 0.0001f)) {
			return false;
		}
	}
	return true;
}

class Circle {
	vec2 center;
	float radius;
	float angle1;
	float angle2;
public:
	Circle(vec2 c, float r) {
		this->center = c;
		this->radius = r;
	}
	void create() {
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)
		vec2 vertices[circleDefinition];
		for (int i = 0; i < circleDefinition; i++) {
			float fi = i * 2 * M_PI / circleDefinition;
			vertices[i] = vec2(this->center.x + this->radius * cosf(fi), this->center.y + this->radius * sinf(fi));
		}
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * circleDefinition,  // # bytes
			vertices,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed
	};
	void drawCircle() {
		// Set color to (0, 1, 0) = green
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, circleRed, circleGreen, circleBlue); // 3 floats

		float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
								  0, 1, 0, 0,    // row-major!
								  0, 0, 1, 0,
								  0, 0, 0, 1 };

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, circleDefinition /*# Elements*/);
	};
};

class CircleList {
public:
	std::vector<Circle> list;

	CircleList() {
		list.push_back(Circle(vec2(0, 0), 1.0f));
	}

	void addCircle(vec2 c, float r) {
		list.push_back(Circle(c, r));
		for (int i = 0; i < list.size(); i++) {
			list[i].create();
		}
	}
};

class Dots {
	unsigned int vaoCtrlDots, vboCtrlDots;
protected:
	std::vector<vec2> ctrlPoints;
public:
	void create() {

		glGenVertexArrays(1, &vaoCtrlDots);
		glBindVertexArray(vaoCtrlDots);
		glGenBuffers(1, &vboCtrlDots);
		glBindBuffer(GL_ARRAY_BUFFER, vboCtrlDots);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);
	};
	void addPoint(float xCord, float yCord) {
		vec4 dot = vec4(xCord, yCord, 0, 1);
		ctrlPoints.push_back(vec2(dot.x, dot.y));
	};
	vec2 getPoint(int index) {
		return ctrlPoints[index];
	};
	void drawDots() {
		if (ctrlPoints.size() > 0) {
			glBindVertexArray(vaoCtrlDots);
			glBindBuffer(GL_ARRAY_BUFFER, vboCtrlDots);
			glBufferData(GL_ARRAY_BUFFER, ctrlPoints.size() * sizeof(vec2), &ctrlPoints[0], GL_DYNAMIC_DRAW);
			gpuProgram.setUniform(vec3(1, 1, 1), "color");
			glPointSize(6.0f);
			glDrawArrays(GL_POINTS, 0, ctrlPoints.size());
		}
	};
};

class EarClass {
	unsigned int vaoEar, vboEar;
	std::vector<vec2> vectorArrayForEar;
	std::vector<vec2> vectorArrayForEarSecond;
public:
	void add(vec2 addThis) {
		if (checkIfUnique(vectorArrayForEar, addThis)) {
			vectorArrayForEar.push_back(addThis);
		}
		else {
			vectorArrayForEar.push_back(addThis);
		}
	}
	int getSize() {
		return vectorArrayForEar.size();
	}
	vec2 get(int id) {
		return vectorArrayForEar.at(id);
	}
	void deleteFromArray(int idx) {
		vectorArrayForEar.erase(vectorArrayForEar.begin() + idx);
	}
	void copyTheArray() {
		vectorArrayForEarSecond = vectorArrayForEar;
	}
	bool testIntersect(vec2 a, vec2 b, vec2 t, vec2 s) {
		float x = ((t.y - s.y) * (a.x - t.x) + (s.x - t.x) * (a.y - t.y)) / ((s.x - t.x) * (a.y - b.y) - (a.x - b.x) * (s.y - t.y));
		float y = ((a.y - b.y) * (a.x - t.x) + (b.x - a.x) * (a.y - t.y)) / ((s.x - t.x) * (a.y - b.y) - (a.x - b.x) * (s.y - t.y));
		return (0 < x && x < 1 && 0 < y && y < 1);
	}
	bool testDiagonal(vec2 a1, vec2 a2, vec2 b1, vec2 b2) {
		int count = 0;
		for (float i = 0.0; i < 1.0; i += 0.01) {
			if (testIntersect(a1, a2, b1, i)) {
				count++;
			}
		}
		return count;
	}
	std::vector<vec2> triangulateFunc() {
		std::vector<vec2> tempVec;
		copyTheArray();
		while (vectorArrayForEarSecond.size() > 3) {
			for (int i = 1; i < vectorArrayForEarSecond.size() - 1; i++) {
				int metszesek = 0;
				vec2 v0 = vectorArrayForEarSecond[i - 1];
				vec2 v1 = vectorArrayForEarSecond[i];
				vec2 v2 = vectorArrayForEarSecond[i + 1];
				vec2 felezo = vec2((v0.x + v2.x) / 2, (v0.y + v2.y) / 2);
				if (testIntersect(v0, v1, p1, felezo)) {
					break;
				}
				else {
					vec2 inf = vec2(1, 1);
					for (int r = 0; r < vectorArrayForEarSecond.size(); r++) {
						int countOfSections = testDiagonal(v0, v1, p1, felezo);
						if (countOfSections % 2 == 1) {
							tempVec.push_back(vectorArrayForEarSecond[i - 1]);
							tempVec.push_back(vectorArrayForEarSecond[0]);
							tempVec.push_back(vectorArrayForEarSecond[i + 1]);
							vectorArrayForEarSecond.erase(vectorArrayForEarSecond.begin() + i);
						}
					}
				}
			}
		}
		tempVec.push_back(vectorArrayForEarSecond[0]);
		tempVec.push_back(vectorArrayForEarSecond[1]);
		tempVec.push_back(vectorArrayForEarSecond[2]);
		return tempVec;
	}

	void create() {
		glGenVertexArrays(1, &vaoEar);	// get 1 vao id
		glGenBuffers(1, &vboEar);	// Generate 1 buffer
	};
	void fill_calc() {
		glBindVertexArray(vaoEar);		// make it active
		glBindBuffer(GL_ARRAY_BUFFER, vboEar);
		vec2 vertices[arcDefinition * 3];
		std::vector<vec2> vertexes;
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//Fülvágó algoritmus elszáll
		/*
		vertexes = triangulateFunc();
		for (int s = 0; s < (arcDefinition * 3); s++) {
			vertices[s] = vertexes[s];
		}
		*/
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * (arcDefinition * 3),  // # bytes
			vertices,	      	// address
			GL_DYNAMIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed

		// Set color to (0, 1, 0) = green
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, fillRed, fillGreen, fillBlue); // 3 floats

		float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
								  0, 1, 0, 0,    // row-major!
								  0, 0, 0, 0,
								  0, 0, 0, 1 };

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

		glBindVertexArray(vaoEar);
		glDrawArrays(GL_TRIANGLES, 0, (arcDefinition * 3));
	};
};

EarClass earCuttingFill;

class ArcClass {
	unsigned int vaoArc, vboArc;
	float aLength = 0.0f;
	float bLength = 0.0f;
	float cLength = 0.0f;
public:
	void create() {
		glGenVertexArrays(1, &vaoArc);
		glGenBuffers(1, &vboArc);
	};
	float getALength() {
		return aLength;
	}
	float getBLength() {
		return bLength;
	}
	float getCLength() {
		return cLength;
	}
	void drawArc(float circleR, float circleX, float circleY, float firstAngle, float secondAngle) {
		glBindVertexArray(vaoArc);
		glBindBuffer(GL_ARRAY_BUFFER, vboArc);
		float minAngle = fminf(firstAngle, secondAngle);
		float maxAngle = fmaxf(firstAngle, secondAngle);
		float max_min_diff = maxAngle - minAngle;
		int size = arcDefinition;
		int index = 0;
		vec2 vertices[arcDefinition];
		if ((maxAngle - minAngle) > M_PI) {
			vec2 last;
			float add = (maxAngle - minAngle) / arcDefinition;
			for (float j = 0; j <= 2 * M_PI; j += add) {
				if ((j >= (maxAngle - add) && j <= 2 * M_PI)) {
					vertices[index] = vec2(circleX + circleR * cosf(j), circleY + circleR * sinf(j));
					index++;
					earCuttingFill.add(vertices[index]);
				}
			}
			for (float j = 0; j <= 2 * M_PI; j += add) {
				if ((j >= 0 && j <= minAngle)) {
					vertices[index] = vec2(circleX + circleR * cosf(j), circleY + circleR * sinf(j));
					last = vec2(circleX + circleR * cosf(j), circleY + circleR * sinf(j));
					index++;
					earCuttingFill.add(vertices[index]);
				}
			}
			for (int k = index; k < arcDefinition; k++) {
				vertices[k] = last;
				index = k;
				earCuttingFill.add(last);
			}
		}
		else {
			float add = (maxAngle - minAngle) / arcDefinition;
			for (float j = 0; j <= 2 * M_PI; j += add) {
				if (j >= minAngle && j <= maxAngle) {
					vertices[index] = vec2(circleX + circleR * cosf(j), circleY + circleR * sinf(j));
					index++;
					earCuttingFill.add(vertices[index]);
				}
			}
		}
		if (printCounter == 0) {
			if (arcCounter == 0) {
				float lenAr = 0.0f;
				for (int i = 0; i < index - 1; i++) {
					lenAr += calculateSirius(vertices[i], vertices[i + 1]);
				}
				printf("Hossz: \n");
				printf("A oldal: %f\n", lenAr);
			}
			if (arcCounter == 1) {
				float lenAr = 0.0f;
				for (int i = 0; i < index - 1; i++) {
					lenAr += calculateSirius(vertices[i], vertices[i + 1]);
				}
				printf("B oldal: %f\n", lenAr);
			}
			if (arcCounter == 2) {
				float lenAr = 0.0f;
				for (int i = 0; i < index - 1; i++) {
					lenAr += calculateSirius(vertices[i], vertices[i + 1]);
				}
				printf("C oldal: %f\n", lenAr);
			}
		}
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(vec2) * arcDefinition,
			vertices,
			GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0,
			2, GL_FLOAT, GL_FALSE,
			0, NULL);

		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, arcRed, arcGreen, arcBlue);

		float MVPtransf[4][4] = { 1, 0, 0, 0,
								  0, 1, 0, 0,
								  0, 0, 0, 0,
								  0, 0, 0, 1 };

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);

		glBindVertexArray(vaoArc);
		glDrawArrays(GL_LINE_STRIP, 0, arcDefinition);
	};
};

CircleList circles;
Dots pontok;
ArcClass arc1;
ArcClass arc2;
ArcClass arc3;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(1.5f);

	for (int i = 0; i < circles.list.size(); i++) {
		circles.list[i].create();
	}

	pontok.create();

	arc1.create();
	arc2.create();
	arc3.create();
	earCuttingFill.create();
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	for (int i = 0; i < circles.list.size(); i++) {
		circles.list[i].drawCircle();
	}

	pontok.drawDots();

	if (clickCounter == 3) {
		vec2 point1 = pontok.getPoint(0);
		vec2 point2 = pontok.getPoint(1);
		vec2 point3 = pontok.getPoint(2);
		p1 = point1;
		p2 = point2;
		p3 = point3;
		vec2 cent1 = calcCenter(point1, point2);
		vec2 cent2 = calcCenter(point2, point3);
		vec2 cent3 = calcCenter(point3, point1);

		float d1 = calcDist(vec2(0, 0), cent1);
		float d2 = calcDist(vec2(0, 0), cent2);
		float d3 = calcDist(vec2(0, 0), cent3);
		float r1 = sqrtf((d1 * d1) - (1 * 1));
		float r2 = sqrtf((d2 * d2) - (1 * 1));
		float r3 = sqrtf((d3 * d3) - (1 * 1));

		float angle11 = atan2f((point1.y - cent1.y), (point1.x - cent1.x));
		float angle12 = atan2f((point2.y - cent1.y), (point2.x - cent1.x));
		float min1 = fminf(angle11, angle12);
		float max1 = fmaxf(angle11, angle12);

		float angle21 = atan2f((point2.y - cent2.y), (point2.x - cent2.x));
		float angle22 = atan2f((point3.y - cent2.y), (point3.x - cent2.x));
		float min2 = fminf(angle21, angle22);
		float max2 = fmaxf(angle21, angle22);

		float angle31 = atan2f((point3.y - cent3.y), (point3.x - cent3.x));
		float angle32 = atan2f((point1.y - cent3.y), (point1.x - cent3.x));
		float min3 = fminf(angle31, angle32);
		float max3 = fmaxf(angle31, angle32);
		if (min1 < 0) {
			min1 += 2 * M_PI;
		}
		if (min2 < 0) {
			min2 += 2 * M_PI;
		}
		if (min3 < 0) {
			min3 += 2 * M_PI;
		}
		if (max1 < 0) {
			max1 += 2 * M_PI;
		}
		if (max2 < 0) {
			max2 += 2 * M_PI;
		}
		if (max3 < 0) {
			max3 += 2 * M_PI;
		}

		arc1.drawArc(r1, cent1.x, cent1.y, min1, max1);
		arcCounter++;
		arc2.drawArc(r2, cent2.x, cent2.y, min2, max2);
		arcCounter++;
		arc3.drawArc(r3, cent3.x, cent3.y, min3, max3);

		aAngle = calculateInnerAngle(cent1, cent2, point2);
		bAngle = calculateInnerAngle(cent2, cent3, point3);
		cAngle = calculateInnerAngle(cent3, cent1, point1);
		if (printCounter == 0) {
			printf("Szogek: \n");
			printf("Alpha: %f\nBeta: %f\nGamma: %f\nSumma: %f\n", aAngle * radConv, bAngle * radConv, cAngle * radConv, ((aAngle * radConv) + (bAngle * radConv) + (cAngle * radConv)));
			printCounter = 1;
		}
		earCuttingFill.fill_calc();
	}
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onMouse(int button, int state, int pX, int pY) {
	float coordX = 2.0f * pX / windowWidth - 1;
	float coordY = 1.0f - 2.0f * pY / windowHeight;
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		pontok.addPoint(coordX, coordY);
		clickCounter += 1;
		//printf("Click: %d \n", clickCounter);
		//printf("x: %f, y: %f \n", coordX, coordY);
		glutPostRedisplay();
	}
}

void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
