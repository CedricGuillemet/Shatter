#include "Shatter.h"
#include <math.h>
#include <float.h>
#include <assert.h>
#include <memory.h>

struct Vec4
{
public:
	Vec4(const Vec4& other) : x(other.x), y(other.y), z(other.z), w(other.w)
	{
	}
	Vec4()
	{
	}
	Vec4(float _x, float _y, float _z = 0.f, float _w = 0.f) : x(_x), y(_y), z(_z), w(_w)
	{
	}
	Vec4(int _x, int _y, int _z = 0, int _w = 0) : x((float)_x), y((float)_y), z((float)_z), w((float)_w)
	{
	}
	Vec4(float v) : x(v), y(v), z(v), w(v)
	{
	}

	void Lerp(const Vec4& v, float t)
	{
		x += (v.x - x) * t;
		y += (v.y - y) * t;
		z += (v.z - z) * t;
		w += (v.w - w) * t;
	}
	void LerpColor(const Vec4& v, float t)
	{
		for (int i = 0; i < 4; i++)
			(*this)[i] = sqrtf(((*this)[i] * (*this)[i]) * (1.f - t) + (v[i] * v[i]) * (t));
	}
	void Lerp(const Vec4& v, const Vec4& v2, float t)
	{
		*this = v;
		Lerp(v2, t);
	}

	inline void Set(float v)
	{
		x = y = z = w = v;
	}
	inline void Set(float _x, float _y, float _z = 0.f, float _w = 0.f)
	{
		x = _x;
		y = _y;
		z = _z;
		w = _w;
	}

	inline Vec4& operator-=(const Vec4& v)
	{
		x -= v.x;
		y -= v.y;
		z -= v.z;
		w -= v.w;
		return *this;
	}
	inline Vec4& operator+=(const Vec4& v)
	{
		x += v.x;
		y += v.y;
		z += v.z;
		w += v.w;
		return *this;
	}
	inline Vec4& operator*=(const Vec4& v)
	{
		x *= v.x;
		y *= v.y;
		z *= v.z;
		w *= v.w;
		return *this;
	}
	inline Vec4& operator*=(float v)
	{
		x *= v;
		y *= v;
		z *= v;
		w *= v;
		return *this;
	}

	inline Vec4 operator*(float f) const;
	inline Vec4 operator-() const;
	inline Vec4 operator-(const Vec4& v) const;
	inline Vec4 operator+(const Vec4& v) const;
	inline Vec4 operator*(const Vec4& v) const;

	inline const Vec4& operator+() const
	{
		return (*this);
	}
	inline float Length() const
	{
		return sqrtf(x * x + y * y + z * z);
	};
	inline float LengthSq() const
	{
		return (x * x + y * y + z * z);
	};
	inline Vec4 Normalize()
	{
		(*this) *= (1.f / Length() + FLT_EPSILON);
		return (*this);
	}
	inline Vec4 Normalize(const Vec4& v)
	{
		this->Set(v.x, v.y, v.z, v.w);
		this->Normalize();
		return (*this);
	}
	inline int LongestAxis() const
	{
		int res = 0;
		res = (fabsf((*this)[1]) > fabsf((*this)[res])) ? 1 : res;
		res = (fabsf((*this)[2]) > fabsf((*this)[res])) ? 2 : res;
		return res;
	}
	inline void Cross(const Vec4& v)
	{
		Vec4 res;
		res.x = y * v.z - z * v.y;
		res.y = z * v.x - x * v.z;
		res.z = x * v.y - y * v.x;

		x = res.x;
		y = res.y;
		z = res.z;
		w = 0.f;
	}
	inline void Cross(const Vec4& v1, const Vec4& v2)
	{
		x = v1.y * v2.z - v1.z * v2.y;
		y = v1.z * v2.x - v1.x * v2.z;
		z = v1.x * v2.y - v1.y * v2.x;
		w = 0.f;
	}
	inline float Dot(const Vec4& v) const
	{
		return (x * v.x) + (y * v.y) + (z * v.z) + (w * v.w);
	}

	void IsMaxOf(const Vec4& v)
	{
		x = (v.x > x) ? v.x : x;
		y = (v.y > y) ? v.y : y;
		z = (v.z > z) ? v.z : z;
		w = (v.w > w) ? v.z : w;
	}
	void IsMinOf(const Vec4& v)
	{
		x = (v.x > x) ? x : v.x;
		y = (v.y > y) ? y : v.y;
		z = (v.z > z) ? z : v.z;
		w = (v.w > w) ? z : v.w;
	}

	bool IsInside(const Vec4& min, const Vec4& max) const
	{
		if (min.x > x || max.x < x || min.y > y || max.y < y || min.z > z || max.z < z)
			return false;
		return true;
	}

	Vec4 Symetrical(const Vec4& v) const
	{
		Vec4 res;
		float dist = SignedDistanceTo(v);
		res = v;
		res -= (*this) * dist * 2.f;

		return res;
	}
	/*void transform(const Mat4x4& matrix);
	void transform(const Vec4 & s, const Mat4x4& matrix);
	*/
	//void TransformVector(const Mat4x4& matrix);
	//void TransformPoint(const Mat4x4& matrix);
	/*
	void TransformVector(const Vec4& v, const Mat4x4& matrix)
	{
		(*this) = v;
		this->TransformVector(matrix);
	}
	void TransformPoint(const Vec4& v, const Mat4x4& matrix)
	{
		(*this) = v;
		this->TransformPoint(matrix);
	}
	*/
	// quaternion slerp
	// void slerp(const Vec4 &q1, const Vec4 &q2, float t );

	inline float SignedDistanceTo(const Vec4& point) const;
	float& operator[](size_t index)
	{
		return ((float*)&x)[index];
	}
	const float& operator[](size_t index) const
	{
		return ((float*)&x)[index];
	}

	float x, y, z, w;
};

inline Vec4 Vec4::operator*(float f) const
{
	return Vec4(x * f, y * f, z * f, w * f);
}
inline Vec4 Vec4::operator-() const
{
	return Vec4(-x, -y, -z, -w);
}
inline Vec4 Vec4::operator-(const Vec4& v) const
{
	return Vec4(x - v.x, y - v.y, z - v.z, w - v.w);
}
inline Vec4 Vec4::operator+(const Vec4& v) const
{
	return Vec4(x + v.x, y + v.y, z + v.z, w + v.w);
}
inline Vec4 Vec4::operator*(const Vec4& v) const
{
	return Vec4(x * v.x, y * v.y, z * v.z, w * v.w);
}
inline float Vec4::SignedDistanceTo(const Vec4& point) const
{
	return (point.Dot(Vec4(x, y, z))) - w;
}

inline Vec4 Normalized(const Vec4& v)
{
	Vec4 res;
	res = v;
	res.Normalize();
	return res;
}
inline Vec4 Cross(const Vec4& v1, const Vec4& v2)
{
	Vec4 res;
	res.x = v1.y * v2.z - v1.z * v2.y;
	res.y = v1.z * v2.x - v1.x * v2.z;
	res.z = v1.x * v2.y - v1.y * v2.x;
	res.w = 0.f;
	return res;
}

inline Vec3 Cross(const Vec3& v1, const Vec3& v2)
{
	Vec3 res;
	res.x = v1.y * v2.z - v1.z * v2.y;
	res.y = v1.z * v2.x - v1.x * v2.z;
	res.z = v1.x * v2.y - v1.y * v2.x;
	return res;
}

inline float Dot(const Vec4& v1, const Vec4& v2)
{
	return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
}

Vec4 BuildPlan(const Vec4& p_point1, const Vec4& p_normal)
{
	Vec4 normal, res;
	normal.Normalize(p_normal);
	res.w = normal.Dot(p_point1);
	res.x = normal.x;
	res.y = normal.y;
	res.z = normal.z;
	return res;
}

Vec3 Lerp(const Vec3& A, const Vec3& B, float t)
{
	return {
		A.x + (B.x - A.x) * t,
		A.y + (B.y - A.y) * t,
		A.z + (B.z - A.z) * t
	};
}

float SquaredDistance(const Vec3& A, const Vec3& B)
{
	return (A.x - B.x) * (A.x - B.x) +
		(A.y - B.y) * (A.y - B.y) +
		(A.z - B.z) * (A.z - B.z);
}

Vec3 NormDirection(const Vec3& A, const Vec3& B)
{
	Vec3 dir = {
		B.x - A.x,
		B.y - A.y,
		B.z - A.z
	};

	float invLength = 1.f / sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
	dir.x *= invLength;
	dir.y *= invLength;
	dir.z *= invLength;
	return dir;
}

Vec3 Normalized(const Vec3& vector)
{
	Vec3 dir = vector;
	float invLength = 1.f / sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
	dir.x *= invLength;
	dir.y *= invLength;
	dir.z *= invLength;
	return dir;
}

Vec3 TriangleNormal(const Vec3 pos[3])
{
	Vec3 dirA = NormDirection(pos[0], pos[1]);
	Vec3 dirB = NormDirection(pos[0], pos[2]);
	Vec3 norm = Normalized(Cross(dirA, dirB));
	return norm;
}

float Dot(const Vec3& A, const Vec3& B)
{
	return A.x * B.x + A.y * B.y + A.z * B.z;
}

uint16_t scratchIndexA[65536];
Vec3 scratchPosA[65536];

uint16_t scratchIndexB[65536];
Vec3 scratchPosB[65536];

template<typename T> void Swap(T& A, T& B)
{
	T temp{ A };
	A = B;
	B = temp;
}

void ComputeBBox(const Vec3* positions, size_t positionCount, Vec3& bbMin, Vec3& bbMax)
{
	bbMin = { FLT_MAX, FLT_MAX, FLT_MAX };
	bbMax = { -FLT_MAX, -FLT_MAX, -FLT_MAX };

	for (size_t i = 0; i < positionCount; i++)
	{
		const Vec3 position = positions[i];
		bbMin.x = (bbMin.x < position.x) ? bbMin.x : position.x;
		bbMin.y = (bbMin.y < position.y) ? bbMin.y : position.y;
		bbMin.z = (bbMin.z < position.z) ? bbMin.z : position.z;

		bbMax.x = (bbMax.x > position.x) ? bbMax.x : position.x;
		bbMax.y = (bbMax.y > position.y) ? bbMax.y : position.y;
		bbMax.z = (bbMax.z > position.z) ? bbMax.z : position.z;
	}
}

template<int count>uint8_t GetSides(const Vec3* pos, const Vec4& plan, float* distances, int& onPlanCount, int* onPlanIndex )
{
	onPlanCount = 0;
	uint8_t ret = 0;
	for (int i = 0; i < count; i++)
	{
		distances[i] = plan.SignedDistanceTo(Vec4(pos[i].x, pos[i].y, pos[i].z));
	}

	// get dominant sign
	int positive = 0;
	int negative = 0;

	for (int i = 0; i < count; i++)
	{
		if (distances[i] > FLT_EPSILON)
		{
			positive ++;
		}
		if (distances[i] < -FLT_EPSILON)
		{
			negative ++;
		}
	}

	for (int i = 0; i < count; i++)
	{
		if (fabsf(distances[i]) < FLT_EPSILON)
		{
			distances[i] = FLT_EPSILON * ((positive>negative) ? 1.f : -1.f);
			onPlanIndex[onPlanCount] = i;
			onPlanCount ++;
		}
	}
	assert(onPlanCount != 3); // TODO : handle case all on plane

	for (int i = 0; i < count; i++)
	{
		ret |= (distances[i] > 0.f) ? (1 << i) : 0;
	}
	return ret;
}

void SortSegments(std::vector<Segment>& segments, size_t currentIndex, size_t& sortedSegmentIndex, size_t* sortedSegments, bool* sortedSegmentUsed)
{
	Vec3 startPosA = segments[currentIndex].mA;
	Vec3 endPosA = segments[currentIndex].mB;
	float bestSqDistance = FLT_MAX;
	int bestIndex = -1;
	for (size_t index = 0; index < segments.size(); index++)
	{
		if (index == currentIndex || sortedSegmentUsed[index])
		{
			continue;
		}
		Vec3 startPosB = segments[index].mA;
		Vec3 endPosB = segments[index].mB;
		float sqDistance = SquaredDistance(endPosA, startPosB);
		if (sqDistance < bestSqDistance)
		{
			bestSqDistance = sqDistance;
			bestIndex = int(index);
		}
	}

	if (bestIndex>0)
	{
		sortedSegments[sortedSegmentIndex++] = bestIndex;
		sortedSegmentUsed[bestIndex] = true;
		SortSegments(segments, bestIndex, sortedSegmentIndex, sortedSegments, sortedSegmentUsed);
	}
}

void SortSegments(std::vector<Segment>& segments)
{
	assert(segments.size() < 1024);
	size_t sortedSegments[1024];
	size_t sortedSegmentIndex = 1;
	bool sortedSegmentUsed[1024] = {};

	sortedSegmentUsed[0] = true;
	sortedSegments[0] = 0;
	SortSegments(segments, 0, sortedSegmentIndex, sortedSegments, sortedSegmentUsed);
	if (sortedSegmentIndex != segments.size())
	{
		segments.clear();
		return;
	}
	std::vector<Segment> res;
	res.resize(segments.size());
	for (size_t i = 0; i < sortedSegmentIndex; i++)
	{
		res[i] = segments[sortedSegments[i]];
	}
	segments = res;
}

void ClipMesh(const Vec3* positions, const unsigned int positionCount, const uint16_t* indices, const unsigned int indexCount,
	Vec3* positionsOut, unsigned int& positionCountOut, uint16_t* indicesOut, unsigned int& indexCountOut,
	std::vector<Segment>& segmentsOut,
	const Vec4 plan)
{
	indexCountOut = 0;
	positionCountOut = 0;
	std::vector<Segment> segments = segmentsOut;
	for (unsigned int i = 0; i < indexCount; i += 3)
	{
		Vec3 trianglePos[3];
		for (int t = 0; t < 3; t++)
		{
			const uint16_t index = indices[i + t];
			trianglePos[t] = positions[index];
		}

		float distances[3];
		int onPlanCount;
		int onPlanIndex[3];
		const uint8_t sides = GetSides<3>(trianglePos, plan, distances, onPlanCount, onPlanIndex);
		if (sides == 0)
		{
			// append triangle
			for (int t = 0; t < 3; t++)
			{
				positionsOut[positionCountOut] = trianglePos[t];
				indicesOut[indexCountOut++] = positionCountOut;
				positionCountOut++;
			}

			if (onPlanCount == 2)
			{
				assert(SquaredDistance(trianglePos[onPlanIndex[0]], trianglePos[onPlanIndex[1]]) > FLT_EPSILON);
				if (onPlanIndex[0] == 0 && onPlanIndex[1] == 1)
				{
					segments.push_back({ trianglePos[1], trianglePos[0] });
				}
				else if (onPlanIndex[0] == 0 && onPlanIndex[1] == 2)
				{
					segments.push_back({ trianglePos[0], trianglePos[2] });
				}
				else if (onPlanIndex[0] == 1 && onPlanIndex[1] == 2)
				{
					segments.push_back({ trianglePos[2], trianglePos[1] });
				}
				else
				{
					assert(0);
				}
			}
		}
		else if (sides == 7)
		{
			// skip
		}
		else
		{
			// cut triangle
			Segment currentSegment;
			Vec3 cutPos[4];
			int cutCount = 0;
			for (int t = 0; t < 3; t++)
			{
				const int indexA = t;
				const int indexB = (t + 1) % 3;
				const float distanceA = distances[indexA];
				const float distanceB = distances[indexB];
				const bool isInA = distanceA < 0.f;
				const bool isAOnPlane = fabsf(distanceA) <= FLT_EPSILON;
				const bool isInB = distanceB < 0.f;
				if (isInA && !isAOnPlane)
				{
					cutPos[cutCount++] = trianglePos[t];
				}
				if (isInA != isInB)
				{
					float len = fabsf(distanceB - distanceA);
					float t = fabsf(distanceA) / len;
					const Vec3& posA = trianglePos[indexA];
					const Vec3& posB = trianglePos[indexB];
					const Vec3 cut = Lerp(posA, posB, t);

					if (isInA)
					{
						currentSegment.mB = cut;
					}
					else
					{
						currentSegment.mA = cut;
					}

					cutPos[cutCount] = cut;
					cutCount++;
				}
			}
			segments.push_back(currentSegment);

			// generate triangles
			assert(cutCount == 3 || cutCount == 4);
			static const int localIndex[] = { 0, 1, 2, 0, 2, 3 };

			const int localIndexCount = (cutCount == 3) ? 3 : 6;
			for (int t = 0; t < localIndexCount; t++)
			{
				positionsOut[positionCountOut] = cutPos[localIndex[t]];
				indicesOut[indexCountOut++] = positionCountOut;
				positionCountOut++;
			}
		}
	}
	segmentsOut = segments;
}

std::vector<Vec3> SegmentsToVertices(const std::vector<Segment>& shape)
{
	std::vector<Vec3> res;
	if (shape.size() < 3)
	{
		return res;
	}

	Vec3 lastDir = NormDirection(shape.back().mA, shape.back().mB);
	for (size_t i = 0; i < shape.size(); i++)
	{
		Vec3 currentDir = NormDirection(shape[i].mA, shape[i].mB);
		float dt = Dot(currentDir, lastDir);
		lastDir = currentDir;

		if (dt < (1.f-FLT_EPSILON))
		{
			res.push_back(shape[i].mA);
		}
	}

	return res;
}

void ClipSegments(std::vector<Segment>& segments, const Vec4 plan, std::vector<Segment>& segmentsForPlan)
{
	std::vector<Segment> segmentsOut;
	uint32_t pointOnPlanCount = 0;
	Vec3 pointOnPlanA, pointOnPlanB;
	for (auto& segment : segments)
	{
		float distances[2];
		int onPlanCount;
		int onPlanIndex[3];
		const uint8_t sides = GetSides<2>(&segment.mA, plan, distances, onPlanCount, onPlanIndex);
		if (sides == 0)
		{
			// append segment
			if (onPlanCount != 2)
			{
				segmentsOut.push_back(segment);
			}
			if (onPlanCount == 1)
			{
				if (onPlanIndex[0] == 0)
				{
					pointOnPlanB = segment.mA;
				}
				else
				{
					pointOnPlanA = segment.mB;
				}
				pointOnPlanCount++;
			}
		}
		else if (sides == 3)
		{
			// skip
		}
		else
		{
			const float distanceA = distances[0];
			const float distanceB = distances[1];
			const bool isInA = distanceA < 0.f;
			const bool isInB = distanceB < 0.f;
			if (isInA != isInB)
			{
				float len = fabsf(distanceB - distanceA);
				float t = fabsf(distanceA) / len;
				const Vec3& posA = segment.mA;
				const Vec3& posB = segment.mB;
				Vec3 cutPos = Lerp(posA, posB, t);
				if (isInA)
				{
					segmentsOut.push_back({ posA, cutPos });
					pointOnPlanA = cutPos;
				}
				else
				{
					segmentsOut.push_back({ cutPos, posB });
					pointOnPlanB = cutPos;
				}
				pointOnPlanCount++;
				assert(pointOnPlanCount < 3);
			}
		}
	}
	assert(pointOnPlanCount == 0 || pointOnPlanCount == 2);
	if (pointOnPlanCount == 2)
	{
		segmentsOut.push_back({ pointOnPlanA, pointOnPlanB });
		segmentsForPlan.push_back({ pointOnPlanB, pointOnPlanA });
	}
	segments = segmentsOut;
}

void ShapeToTriangles(const std::vector<Vec3>& shape, Vec3* positionsOut, unsigned int& positionCountOut, uint16_t* indicesOut, unsigned int& indexCountOut)
{
	if (shape.size() > 2)
	{
		Vec3 posOrigin = shape[0];
		for (size_t index = 1; index < (shape.size() - 1); index++)
		{
			Vec3 posA = shape[index];
			Vec3 posB = shape[index + 1];

			positionsOut[positionCountOut] = posOrigin;
			indicesOut[indexCountOut++] = positionCountOut;
			positionCountOut++;

			positionsOut[positionCountOut] = posA;
			indicesOut[indexCountOut++] = positionCountOut;
			positionCountOut++;

			positionsOut[positionCountOut] = posB;
			indicesOut[indexCountOut++] = positionCountOut;
			positionCountOut++;
		}
	}
}

ShatterMeshes Shatter(const Vec3* impacts, const unsigned int impactCount, const Vec3* positions, const unsigned int positionCount, const uint16_t* indices, const unsigned int indexCount)
{
	ShatterMeshes result{};

	for (size_t impactIndexA = 0; impactIndexA < impactCount; impactIndexA++)
	{
		memcpy(scratchIndexA, indices, indexCount * sizeof(uint16_t));
		memcpy(scratchPosA, positions, positionCount * sizeof(Vec3));

		Vec3* positionsIn = scratchPosA;
		unsigned int positionCountIn = positionCount;
		uint16_t* indicesIn = scratchIndexA;
		unsigned int indexCountIn = indexCount;

		Vec3* positionsOut = scratchPosB;
		uint16_t* indicesOut = scratchIndexB;
		unsigned int positionCountOut, indexCountOut;

		std::vector<std::vector<Segment>> meshSegments;
		meshSegments.resize(impactCount);
		for (size_t impactIndexB = 0; impactIndexB < impactCount; impactIndexB++)
		{
			if (impactIndexA == impactIndexB)
			{
				continue;
			}
			// plan
			Vec4 impactA(impacts[impactIndexA].x, impacts[impactIndexA].y, impacts[impactIndexA].z);
			Vec4 impactB(impacts[impactIndexB].x, impacts[impactIndexB].y, impacts[impactIndexB].z);
			Vec4 plan = BuildPlan((impactA + impactB) * 0.5f, (impactB - impactA).Normalize());

			for (size_t shapeIndex = 0;shapeIndex < impactIndexB; shapeIndex++)
			{
				auto& segments = meshSegments[shapeIndex];
				if (segments.size() > 2)
				{
					ClipSegments(segments, plan, meshSegments[impactIndexB]);
				}
			}
			ClipMesh(positionsIn, positionCountIn, indicesIn, indexCountIn,
				positionsOut, positionCountOut, indicesOut, indexCountOut,
				meshSegments[impactIndexB],
				//segments,
				plan);

			Swap(positionsIn, positionsOut);
			Swap(indicesIn, indicesOut);
			indexCountIn = indexCountOut;
			positionCountIn = positionCountOut;
		}

		// from seg to triangles
		for (auto& segments : meshSegments)
		{
			if (segments.size() > 2)
			{
				SortSegments(segments);
				auto vertices = SegmentsToVertices(segments);
				ShapeToTriangles(vertices, positionsIn, positionCountIn, indicesIn, indexCountIn);
			}
		}

		if (indexCountIn && positionCountIn)
		{
			result.mShatteredMeshCount++;
			auto newShatteredMeshes = new ShatteredMesh[result.mShatteredMeshCount];
			if (result.mShatteredMeshCount > 1)
			{
				memcpy(newShatteredMeshes, result.mShatteredMeshes, sizeof(ShatteredMesh) * (result.mShatteredMeshCount - 1));
			}
			result.mShatteredMeshes = newShatteredMeshes;
			auto& mesh = result.mShatteredMeshes[result.mShatteredMeshCount-1];

			mesh.mIndexCount = indexCountIn;
			mesh.mPositionCount = positionCountIn;
			mesh.mIndices = new uint16_t[indexCountIn];
			memcpy(mesh.mIndices, indicesIn, indexCountIn * sizeof(uint16_t));
			mesh.mPositions = new Vec3[positionCountIn];
			memcpy(mesh.mPositions, positionsIn, positionCountIn * sizeof(Vec3));

			ComputeBBox(mesh.mPositions, positionCountIn, mesh.mBBMin, mesh.mBBMax);
			auto center = mesh.GetBBoxCenter();
			for (size_t i = 0; i < positionCountIn; i++)
			{
				mesh.mPositions[i].x -= center.x;
				mesh.mPositions[i].y -= center.y;
				mesh.mPositions[i].z -= center.z;
			}
			/*
			mesh.mShapes.resize(meshSegments.size());
			for (size_t i = 0;i< meshSegments.size();i++)
			{
				if (meshSegments[i].size() > 2)
				{
					mesh.mShapes[i].mSegments = meshSegments[i];
				}
			}
			*/
		}
	}
	return result;
}

void ComputeExplosion(const Vec3& explosionSource, const Vec3& explosionDirection, const Vec3& debrisPosition, Vec3& debrisDirection, float& debrisForce, Vec3& debrisTorqueAxis)
{
	debrisDirection = NormDirection(explosionSource, debrisPosition);
	debrisForce = Dot(explosionDirection, debrisDirection);
	debrisTorqueAxis = Normalized(Cross(explosionDirection, debrisDirection));
}



void rotationAxis(const Vec3& axis, float angle, float *m)
{
	float length2 = axis.x * axis.x + axis.y * axis.y + axis.z * axis.z;
	/*
	if (length2 < FLOAT_EPSILON)
	{
		identity();
		return;
	}
	*/
	Vec3 n = axis;// * (1.f / sqrtf(length2));
	float s = sinf(angle);
	float c = cosf(angle);
	float k = 1.f - c;

	float xx = n.x * n.x * k + c;
	float yy = n.y * n.y * k + c;
	float zz = n.z * n.z * k + c;
	float xy = n.x * n.y * k;
	float yz = n.y * n.z * k;
	float zx = n.z * n.x * k;
	float xs = n.x * s;
	float ys = n.y * s;
	float zs = n.z * s;

	m[0 * 4 + 0] = xx;
	m[0 * 4 + 1] = xy + zs;
	m[0 * 4 + 2] = zx - ys;
	m[0 * 4 + 3] = 0.f;
	m[1 * 4 + 0] = xy - zs;
	m[1 * 4 + 1] = yy;
	m[1 * 4 + 2] = yz + xs;
	m[1 * 4 + 3] = 0.f;
	m[2 * 4 + 0] = zx + ys;
	m[2 * 4 + 1] = yz - xs;
	m[2 * 4 + 2] = zz;
	m[2 * 4 + 3] = 0.f;
	m[3 * 4 + 0] = 0.f;
	m[3 * 4 + 1] = 0.f;
	m[3 * 4 + 2] = 0.f;
	m[3 * 4 + 3] = 1.f;
}
