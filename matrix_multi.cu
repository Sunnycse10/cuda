#include<iostream>
#include<cstdlib>
#include<vector>
#include<algorithm>

using namespace std;


__global__ void matrix_multi( int *a, int *b, int *c, int n)
{
    //calculating row and column for every matrices
    int row = blockDim.y*blockIdx.y+ threadIdx.y;
    int col = blockDim.x*blockIdx.x+ threadIdx.x;

    if(row<n && col<n)
    {
        int tmp =0;
        for(int k=0; k<n; k++)
        {
            tmp+= a[n*row+k] * b[k*n+col];
        }
        c[n*row+col] = tmp;
    }

}


//initializing square matrix with random integers
void matrix_init(vector<int>&a, int n)
{
   for(int i=0;i<n*n;i++)
   {
    a[i]= rand()%10;
   }
}

void print_matrix(vector<int> &a, int n)
{
    for(int i=0;i<n*n;i++)
    {
     cout<<a[i]<<" ";
    }
}


int main()
{
    int n=3;//width of the square matrix
    size_t bytes = n*n*sizeof(int);
    vector<int> h_a(n*n), h_b(n*n), h_c(n*n);
    matrix_init(h_a,n);
    matrix_init(h_b,n);
    //cudaMallocMananged(&a,bytes);
    //cudaMallocMananged(&b,bytes);
    //cudaMallocMananged(&c,bytes);
    int *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,bytes);
    cudaMalloc(&d_b,bytes);
    cudaMalloc(&d_c,bytes);
    cudaMemcpy(d_a,h_a.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b.data(),bytes,cudaMemcpyHostToDevice);
    //cudaMemcpy(d_c,h_c,bytes,cudaMemcpyHostToDevice);
    int threads = 32;
    int blocks = (n+threads-1)/threads;
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);
    //launch our kernel
    matrix_multi<<<BLOCKS,THREADS>>>(d_a,d_b,d_c,n);
    cudaMemcpy(h_c.data(),d_c,bytes,cudaMemcpyDeviceToHost);
    cout<<"matrix a"<<endl;
    print_matrix(h_a,n);
    cout<<"matrix b"<<endl;
    print_matrix(h_b,n);
    cout<<"matrix c"<<endl;
    print_matrix(h_c,n);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}