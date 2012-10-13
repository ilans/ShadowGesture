// ShadowGesture.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "ShadowGesture.h"
#include "Facerec.h"

int _tmain(int argc, _TCHAR* argv[])
{
	ShadowGesture* sg = new ShadowGesture();
	//sg->capture();
	//sg->extract();
	//sg->vectorize();
	//sg->trainHMM();
	//sg->testHMM("../Data/test_sequense.yml");
	sg->FindConvexityDefects();

	//Facerec facerec("../Data/path_labels.csv");

	return 0;
}

