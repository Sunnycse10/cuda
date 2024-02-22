#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <sys/time.h>

using namespace std;

#define totaldegrees 180
#define binsperdegree 4
#define threadsperblock 512
// data for the real galaxies will be read into these arrays
float *ra_real, *decl_real;
// number of real galaxies
int    NoofReal;

// data for the simulated random galaxies will be read into these arrays
float *ra_sim, *decl_sim;
// number of simulated random galaxies
int    NoofSim;

unsigned int *histogramDR, *histogramDD, *histogramRR;
unsigned int *d_histogram;


/*
//calculate histogram

__device__ void calculateHistogram(float x, unsigned int* histogram)
{ 
    unsigned int index = (unsigned int) x*binsperdegree;
    atomicAdd(&histogram[index],1);

}
*/



//calculate DD, DR and RR and update count on histogram 
__global__ void calculateDD(float *d_ra_real, float *d_decl_real, float *d_ra_sim, float *d_decl_sim, unsigned int *hist_dd_result, unsigned int *hist_dr_result,unsigned int *hist_rr_result, int NoofReal)
{

    int Xid = blockDim.x*blockIdx.x + threadIdx.x;
    if(Xid<NoofReal)
    {
        for(int k=0;k<NoofReal;k++)
        {
            float dd = sin(d_decl_real[Xid])*sin(d_decl_real[k]) + cos(d_decl_real[Xid])*cos(d_decl_real[k])*cos(d_ra_real[Xid]-d_ra_real[k]);
            float dr = sin(d_decl_real[Xid])*sin(d_decl_sim[k]) + cos(d_decl_real[Xid])*cos(d_decl_sim[k])*cos(d_ra_real[Xid]-d_ra_sim[k]);
            float rr = sin(d_decl_sim[Xid])*sin(d_decl_sim[k]) + cos(d_decl_sim[Xid])*cos(d_decl_sim[k])*cos(d_ra_sim[Xid]-d_ra_sim[k]);
            dd = fminf(1.0f,fmaxf(-1.0f,dd));
            dr = fminf(1.0f,fmaxf(-1.0f,dr));
            rr = fminf(1.0f,fmaxf(-1.0f,rr));
            dd = acos(dd)*180.0f/3.141592654f;
            dr = acos(dr)*180.0f/3.141592654f;
            rr = acos(rr)*180.0f/3.141592654f;
            atomicAdd(&hist_dd_result[(int)(dd*4.0f)],1U);
            atomicAdd(&hist_dr_result[(int)(dr*4.0f)],1U);
            atomicAdd(&hist_rr_result[(int)(rr*4.0f)],1U);

        }
        
    }
}
/*
//Calculate DR
__global__ void calculateDR(float *d_ra_real, float *d_decl_real, float *d_ra_sim, float *d_decl_sim, unsigned int *hist_dd_result, unsigned int *hist_dr_result,unsigned int *hist_rr_result, int NoofReal)
{

    int Xid = blockDim.x*blockIdx.x + threadIdx.x;
    if(Xid<NoofReal)
    {
        for(int k=0;k<NoofReal;k++)
        {
            float x = sin(d_decl_real[Xid])*sin(d_decl_real[k]) + cos(d_decl_real[Xid])*cos(d_decl_real[k])*cos(d_ra_real[Xid]-d_ra_real[k]);
            x = fminf(1.0f,fmaxf(-1.0f,x));
            x = acos(x)*180.0f/3.141592654f;
            atomicAdd(&hist_dd_result[(int)(x*4.0f)],1U);

        }
        
    }
}

*/


int main(int argc, char *argv[])
{
   int    i;
   int    noofblocks;
   int    readdata(char *argv1, char *argv2);
   int    getDevice(int deviceno);
   long int histogramDRsum, histogramDDsum, histogramRRsum;
   float w;
   double start, end, kerneltime;
   struct timeval _ttime;
   struct timezone _tzone;
   cudaError_t myError;
   //float mtr = 3.14159/(180*60); //arcmin to radian
   int numberOfBins = totaldegrees*binsperdegree;
    kerneltime = 0.0;
   gettimeofday(&_ttime, &_tzone);
   start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;


   FILE *outfil;

   //if ( argc != 4 ) {printf("Usage: a.out real_data random_data output_data\n");return(-1);}

   if ( getDevice(0) != 0 ) return(-1);

   if ( readdata(argv[1], argv[2]) != 0 ) return(-1);

   /*
   for(int k=0;k<5;k++)
   {
        cout<<ra_real[k]<<" "<<decl_real[k]<<endl;
   }
   */

   //Allocate memory for histograms in cpu

   histogramDD = (unsigned int *) calloc(numberOfBins, sizeof(unsigned int));
   histogramDR = (unsigned int *) calloc(numberOfBins, sizeof(unsigned int));
   histogramRR = (unsigned int *) calloc(numberOfBins, sizeof(unsigned int));

   // allocate memory on the GPU
   float *d_ra_real, *d_decl_real, *d_ra_sim, *d_decl_sim;
   unsigned int *histogramDD_result, *histogramDR_result, *histogramRR_result;
   size_t sizeOfRealGal = NoofReal * sizeof(float);
   size_t sizeOfSimGal = NoofSim * sizeof(float);
   size_t sizeOfHistogram = numberOfBins * sizeof(unsigned int);
   if(cudaMalloc(&d_ra_real,sizeOfRealGal)==cudaSuccess)
        cout<<"ra real malloc is successful"<<endl;
   else
   {
        cout<<"ra real malloc is unsuccessful"<<endl;
        return 0;
   }
   if(cudaMalloc(&d_decl_real,sizeOfRealGal)==cudaSuccess)
        cout<<"decl real malloc is successful"<<endl;
   else
   {
        cout<<"decl real malloc is unsuccessful"<<endl;
        return 0;

   }
   if(cudaMalloc(&d_ra_sim,sizeOfSimGal)==cudaSuccess)
        cout<<"ra sim malloc is successful"<<endl;
   else
   {
        cout<<"ra sim malloc is unsuccessful"<<endl;
        return 0;
   }
   if(cudaMalloc(&d_decl_sim,sizeOfSimGal)==cudaSuccess)
        cout<<"decl sim malloc is successful"<<endl;
   else
   {
        cout<<"decl sim malloc is unsuccessful"<<endl;
        return 0;

   }
   if(cudaMalloc(&histogramDD_result,sizeOfHistogram)==cudaSuccess)
        cout<<"histogramDD malloc is successful"<<endl;
   else
   {
        cout<<"histogramDD malloc is unsuccessful"<<endl;
        return 0;
   }

   if(cudaMalloc(&histogramDR_result,sizeOfHistogram)==cudaSuccess)
        cout<<"histogramDR malloc is successful"<<endl;
   else
   {
        cout<<"histogramDR malloc is unsuccessful"<<endl;
        return 0;
   }
   if(cudaMalloc(&histogramRR_result,sizeOfHistogram)==cudaSuccess)
        cout<<"histogramRR malloc is successful"<<endl;
   else
   {
        cout<<"histogramRR malloc is unsuccessful"<<endl;
        return 0;
   }


   // copy data to the GPU
   if(cudaMemcpy(d_ra_real,ra_real,sizeOfRealGal,cudaMemcpyHostToDevice)==cudaSuccess)
        cout<<"ra real mem copy is successful"<<endl;
   else
   {
        cout<<"ra real mem copy is unsuccessful"<<endl;
        cudaFree(d_ra_real);
        cudaFree(d_decl_real);
        cudaFree(d_ra_sim);
        cudaFree(d_decl_sim);
        cudaFree(histogramDD_result);
        cudaFree(histogramDR_result);
        cudaFree(histogramRR_result);
        return 0;
   } 
   if(cudaMemcpy(d_decl_real,decl_real,sizeOfRealGal, cudaMemcpyHostToDevice) == cudaSuccess)
        cout<<"decl real mem copy is successful"<<endl;
    else
    {
        cout<<"decl real mem copy is unsuccessful"<<endl;
        cudaFree(d_ra_real);
        cudaFree(d_decl_real);
        cudaFree(d_ra_sim);
        cudaFree(d_decl_sim);
        cudaFree(histogramDD_result);
        cudaFree(histogramDR_result);
        cudaFree(histogramRR_result);
        return 0;   
    }
     if(cudaMemcpy(d_ra_sim,ra_sim,sizeOfSimGal,cudaMemcpyHostToDevice)==cudaSuccess)
        cout<<"ra sim mem copy is successful"<<endl;
   else
   {
        cout<<"ra sim mem copy is unsuccessful"<<endl;
        cudaFree(d_ra_real);
        cudaFree(d_decl_real);
        cudaFree(d_ra_sim);
        cudaFree(d_decl_sim);
        cudaFree(histogramDD_result);
        cudaFree(histogramDR_result);
        cudaFree(histogramRR_result);
        return 0;
   } 
   if(cudaMemcpy(d_decl_sim,decl_sim,sizeOfSimGal, cudaMemcpyHostToDevice) == cudaSuccess)
        cout<<"decl sim mem copy is successful"<<endl;
    else
    {
        cout<<"decl sim mem copy is unsuccessful"<<endl;
        cudaFree(d_ra_real);
        cudaFree(d_decl_real);
        cudaFree(d_ra_sim);
        cudaFree(d_decl_sim);
        cudaFree(histogramDD_result);
        cudaFree(histogramDR_result);
        cudaFree(histogramRR_result);
        return 0;   
    }
   if(cudaMemcpy(histogramDD_result,histogramDD, sizeOfHistogram, cudaMemcpyHostToDevice)==cudaSuccess)
        cout<<"histogramDD mem copy is successful"<<endl;
   else
   {
        cout<<"histogramDD mem copy is unsuccessful"<<endl;
        cudaFree(d_ra_real);
        cudaFree(d_decl_real);
        cudaFree(d_ra_sim);
        cudaFree(d_decl_sim);
        cudaFree(histogramDD_result);
        cudaFree(histogramDR_result);
        cudaFree(histogramRR_result);
        return 0;

   }
   if(cudaMemcpy(histogramDR_result,histogramDR, sizeOfHistogram, cudaMemcpyHostToDevice)==cudaSuccess)
        cout<<"histogramDR mem copy is successful"<<endl;
   else
   {
        cout<<"histogramDR mem copy is unsuccessful"<<endl;
        cudaFree(d_ra_real);
        cudaFree(d_decl_real);
        cudaFree(d_ra_sim);
        cudaFree(d_decl_sim);
        cudaFree(histogramDD_result);
        cudaFree(histogramDR_result);
        cudaFree(histogramRR_result);
        return 0;

   }
    if(cudaMemcpy(histogramRR_result,histogramRR, sizeOfHistogram, cudaMemcpyHostToDevice)==cudaSuccess)
        cout<<"histogramRR mem copy is successful"<<endl;
   else
   {
        cout<<"histogramRR mem copy is unsuccessful"<<endl;
        cudaFree(d_ra_real);
        cudaFree(d_decl_real);
        cudaFree(d_ra_sim);
        cudaFree(d_decl_sim);
        cudaFree(histogramDD_result);
        cudaFree(histogramDR_result);
        cudaFree(histogramRR_result);
        return 0;

   }

   // run the kernels on the GPU
   dim3 thread(threadsperblock);
   noofblocks = (NoofReal+threadsperblock-1)/(threadsperblock);
   dim3 blocks(noofblocks);

   calculateDD<<<blocks, thread>>>(d_ra_real,d_decl_real, d_ra_sim, d_decl_sim, histogramDD_result,histogramDR_result, histogramRR_result, NoofReal);

   // copy the results back to the CPU
   cudaMemcpy(histogramDD,histogramDD_result,sizeOfHistogram,cudaMemcpyDeviceToHost);
   cudaMemcpy(histogramDR,histogramDR_result,sizeOfHistogram,cudaMemcpyDeviceToHost);
   cudaMemcpy(histogramRR,histogramRR_result,sizeOfHistogram,cudaMemcpyDeviceToHost);

   // calculate omega values on the CPU

   long unsigned int sumDD=0,sumDR=0,sumRR=0;
   vector<float> omg;//array of the omega values
   printf("degree  omega      hist_DD        hist_DR        hist_RR\n");
   for (int index=0;index<numberOfBins;index++)
   {
     w = (1.0f*histogramDD[index] - 2.0f*histogramDR[index] + 1.0f*histogramRR[index])/(1.0f*histogramRR[index]);
     omg.push_back(w);
     printf("%.2f    %f    %u       %u       %u\n",float(index)/4.0, w,histogramDD[index],histogramDR[index],histogramRR[index] );
     /*
     if(index<5)
     {
          printf("%.2f    %f    %u       %u       %u\n",float(index)/4.0, w,histogramDD[index],histogramDR[index],histogramRR[index] );

     }
     */
     sumDD+=histogramDD[index];
     sumDR+=histogramDR[index];
     sumRR+=histogramRR[index];
     
   }
    printf("sum of DD:%lu        sum of DR:%lu   and sum of RR:%lu\n",sumDD,sumDR,sumRR); 

   


   gettimeofday(&_ttime, &_tzone);
   end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
   kerneltime += end-start;
   cout<<"kernel time is: "<<kerneltime<<endl;
   cudaFree(d_ra_real);
   cudaFree(d_decl_real);
   cudaFree(d_ra_sim);
   cudaFree(d_decl_sim);
   cudaFree(histogramDD_result);
   cudaFree(histogramDR_result);
   cudaFree(histogramRR_result);

   return(0);
}


int readdata(char *argv1, char *argv2)
{
  int i,linecount;
  char inbuf[180];
  double ra, dec, phi, theta, dpi;
  FILE *infil;
                                         
  printf("   Assuming input data is given in arc minutes!\n");
                          // spherical coordinates phi and theta:
                          // phi   = ra/60.0 * dpi/180.0;
                          // theta = (90.0-dec/60.0)*dpi/180.0;

  dpi = acos(-1.0);
  infil = fopen(argv1,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv1);return(-1);}

  // read the number of galaxies in the input file
  int announcednumber;
  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv1);return(-1);}
  linecount =0;
  while ( fgets(inbuf,180,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv1, linecount);
  else 
      {
      printf("   %s does not contain %d galaxies but %d\n",argv1, announcednumber,linecount);
      return(-1);
      }

  NoofReal = linecount;
  ra_real   = (float *)calloc(NoofReal,sizeof(float));
  decl_real = (float *)calloc(NoofReal,sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i = 0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) 
         {
         printf("   Cannot read line %d in %s\n",i+1,argv1);
         fclose(infil);
         return(-1);
         }
      ra_real[i]   = (float)ra/60.0f/180.0f*3.141592654f;
      decl_real[i] = (float)dec/60.0f/180.0f*3.141592654f;
      ++i;
      }

  fclose(infil);

  if ( i != NoofReal ) 
      {
      printf("   Cannot read %s correctly\n",argv1);
      return(-1);
      }

  infil = fopen(argv2,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv2);return(-1);}

  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv2);return(-1);}
  linecount =0;
  while ( fgets(inbuf,80,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv2, linecount);
  else
      {
      printf("   %s does not contain %d galaxies but %d\n",argv2, announcednumber,linecount);
      return(-1);
      }

  NoofSim = linecount;
  ra_sim   = (float *)calloc(NoofSim,sizeof(float));
  decl_sim = (float *)calloc(NoofSim,sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i =0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) 
         {
         printf("   Cannot read line %d in %s\n",i+1,argv2);
         fclose(infil);
         return(-1);
         }
      ra_sim[i]   = (float)ra/60.0f/180.0f*3.141592654f;
      decl_sim[i] = (float)dec/60.0f/180.0f*3.141592654f;
      ++i;
      }

  fclose(infil);

  if ( i != NoofSim ) 
      {
      printf("   Cannot read %s correctly\n",argv2);
      return(-1);
      }

  return(0);
}




int getDevice(int deviceNo)
{

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("   Found %d CUDA devices\n",deviceCount);
  if ( deviceCount < 0 || deviceCount > 128 ) return(-1);
  int device;
  for (device = 0; device < deviceCount; ++device) {
       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, device);
       printf("      Device %s                  device %d\n", deviceProp.name,device);
       printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
       printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
       printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
       printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
       printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
       printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
       printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
       printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
       printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate/1000.0);
       printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
       printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
       printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
       printf("         maxGridSize                   =   %d x %d x %d\n",
                          deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
       printf("         maxThreadsDim in thread block =   %d x %d x %d\n",
                          deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
       printf("         concurrentKernels             =   ");
       if(deviceProp.concurrentKernels==1) printf("     yes\n"); else printf("    no\n");
       printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
       if(deviceProp.deviceOverlap == 1)
       printf("            Concurrently copy memory/execute kernel\n");
       }

    cudaSetDevice(deviceNo);
    cudaGetDevice(&device);
    if ( device != 0 ) printf("   Unable to set device 0, using %d instead",device);
    else printf("   Using CUDA device %d\n\n", device);

return(0);
}

