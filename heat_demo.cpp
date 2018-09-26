/******************************************************************************
Start up demo program for 159735 Assignment 3 Semester 1 2013
******************************************************************************/
#include <iostream>
#include <omp.h>
#include <time.h>

#include "arrayff.hxx"
#include "draw.hxx"

int main(int argc, char* argv[]) 
{
	const float tol = 0.00001;
	const int npix = atoi(argv[1]);
	const int npixx = npix;
	const int npixy = npix;
	const int ntotal = npixx * npixy;
	 
	
	Array<float, 2> h(npixy, npixy), g(npixy, npixx);

	const int nrequired = npixx * npixy;
	const int ITMAX = 100000;

	int iter = 0;
	int nconverged = 0;

	fix_boundaries2(h);
	dump_array<float, 2>(h, "plate0.fit");

	omp_set_num_threads(2);

	clock_t begin = clock();

	int sizePerThread, startpoint, endpoint;

	do {

		#pragma omp parallel 
		{
			sizePerThread = npix / omp_get_num_threads();
			startpoint = omp_get_thread_num() * sizePerThread;
			endpoint = startpoint + sizePerThread;

	    	for (int y = startpoint; y < endpoint; ++y) {
		    	for (int x = 1; x < npixx-1; ++x) {
		    		if(y == 0) continue; 
		    		if(y == npix - 1) continue;
		    		if((y == startpoint) || (y == endpoint-1)){ 
		    			#pragma omp critical
	    				g(y, x) = 0.25 * (h(y, x-1) + h(y, x+1) + h(y-1, x) + h(y+1,x));
	    			}else{
						g(y, x) = 0.25 * (h(y, x-1) + h(y, x+1) + h(y-1, x) + h(y+1,x));
					}
		      	}
		    }
	    }

	    fix_boundaries2(g);

	    nconverged = 0;
	    #pragma omp parallel 
		{
			sizePerThread = npix / omp_get_num_threads();
			startpoint = omp_get_thread_num() * sizePerThread;
			endpoint = startpoint + sizePerThread;

		    for (int y = startpoint; y < endpoint; ++y) {
		    	for (int x = 0; x < npixx; ++x) {
					float dhg = std::fabs(g(y, x) - h(y, x));
					if (dhg < tol){
						#pragma omp critical
						++nconverged;
					}
					h(y, x) = g(y, x);
		      	}
		    }
		}
		std::cout << nconverged << std::endl;
	    ++iter;
	}while (nconverged < nrequired && iter < ITMAX);

  	clock_t end = clock();

	dump_array<float, 2>(h, "plate1.fit");
	std::cout << "Required " << iter << " iterations" << std::endl;

	std::cout << "total time: " << (double)(end - begin) / CLOCKS_PER_SEC << std::endl;
}
