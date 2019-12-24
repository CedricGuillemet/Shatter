#pragma once
#include <stdint.h>
#include <vector>

struct Vec3
{
	float x, y, z;
};

struct Segment
{
	Vec3 mA;
	Vec3 mB;
};

struct Shape
{
	std::vector<Segment> mSegments;
	Vec3 mNormal;
};

struct ShatteredMesh
{
	Vec3* mPositions;
	uint16_t* mIndices;
	Vec3 mBBMin;
	Vec3 mBBMax;
	uint32_t mPositionCount;
	uint32_t mIndexCount;
	std::vector<Shape> mShapes;

	Vec3 GetBBoxCenter() const
	{
		return {
			(mBBMin.x + mBBMax.x) * 0.5f,
			(mBBMin.y + mBBMax.y) * 0.5f,
			(mBBMin.z + mBBMax.z) * 0.5f
		};
	}
};

struct ShatterMeshes
{
	ShatteredMesh* mShatteredMeshes;
	uint32_t mShatteredMeshCount;
};

ShatterMeshes Shatter(const Vec3* impacts, const unsigned int impactCount, const Vec3* positions, const unsigned int positionCount, const uint16_t* indices, const unsigned int indexCount);

void ComputeExplosion(const Vec3& explosionSource, const Vec3& explosionDirection, const Vec3& debrisPosition, Vec3& debrisDirection, float& debrisForce, Vec3& debrisTorqueAxis);
void rotationAxis(const Vec3& axis, float angle, float* m);
