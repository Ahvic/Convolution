#include <opencv2/opencv.hpp>
#include <vector>
#include <omp.h>

//Convolution avec global memory
//les composantes rgb sont traitées indépendements
__global__ void convolution_global(unsigned char *imgPad, int paddedH, int paddedW, float *filter, unsigned char* resultat, int cols, int rows, int filter_size){

  int padSize = (( filter_size-1 ) / 2) * 3;
  paddedW *= 3;

  //La position de notre pixel
  int i = blockIdx.x * blockDim.x + threadIdx.x + padSize;
  int j = blockIdx.y * blockDim.y + threadIdx.y + padSize;

  //On ignore les blocs qui ne sont pas dans l'image
  if( i >= padSize && i < paddedW - padSize && j >= padSize && j < paddedH - padSize ) {
    int pixelPos = (i - padSize ) * cols + (j -padSize);
    unsigned char somme = 0;

    int iterationY = 0;
    for(int y = -padSize; y <= padSize; y+=3){
      int iterationX = 0;
      for(int x = -padSize; x <= padSize; x+=3){
        int iPixelPos = ( i + y ) * paddedH + ( j + x );
        int filtrePos = iterationY * filter_size + iterationX;
        somme += imgPad[iPixelPos] * filter[filtrePos];
        iterationX++;
      }
      iterationY++;
    }

    resultat[pixelPos] = somme;
  }
}

std::vector< float> choixFiltre(int choix, int taille)
{
  switch (choix) {
    //Sobel
    case 1: return {1.0,0.0,-1.0,
                    2.0,0.0,-2.0,
                    1.0,0.0,-1.0};
    //Flou
    case 2: return {0.11,0.11,0.11,
                    0.11,0.11,0.11,
                    0.11,0.11,0.11};
  }

  return {};
}

int main()
{
  //Lit l'image
  cv::Mat h_image = cv::imread( "in.jpg", cv::IMREAD_UNCHANGED );
  int W = h_image.rows;
  int H = h_image.cols;

  //Filtre
  int filter_size = 3;
  //Matrice pour un flou
  std::vector< float > h_filter {1.0,0.0,-1.0,
                                 2.0,0.0,-2.0,
                                 1.0,0.0,-1.0};

  //Padding
  int padSize = ( (filter_size-1) / 2 ) * 3;
  cv::Scalar valeurVide(0, 0, 0);
  copyMakeBorder(h_image, h_image ,padSize ,padSize ,padSize ,padSize ,cv::BORDER_CONSTANT, valeurVide);
  int paddedW = h_image.rows;
  int paddedH = h_image.cols;

  //Allocation sur device et transfert filtre
  float * d_filter;
  int filterSizeByte = filter_size * filter_size * sizeof(float);
  cudaMalloc( &d_filter, filterSizeByte);
  cudaMemcpy(d_filter, h_filter.data(), filterSizeByte, cudaMemcpyHostToDevice);

  //Allocation de l'espace pour l'image pad sur le device et transfert
  unsigned char * d_image;
  int imgPadSizeByte = paddedH * paddedW * 3 * sizeof(uchar);
  cudaMalloc( &d_image, imgPadSizeByte);
  cudaMemcpy(d_image, h_image.data, imgPadSizeByte, cudaMemcpyHostToDevice );

  //Config kernel
  //La gtx 780 qu'on utilise supporte 1024 threads/blocs (= 32 x 32)
  dim3 threadBlock( 32, 32 );
  dim3 grille( ( W * 3 - 1) / threadBlock.x + 1 , ( H - 1 ) / threadBlock.y + 1 );

  //Appel kernel
  unsigned char * d_resultat;
  cudaMalloc( &d_resultat, W * H * 3);

  convolution_global<<< grille, threadBlock >>>( d_image, paddedH, paddedW, d_filter, d_resultat, H, W, filter_size );
  cudaDeviceSynchronize();

  //Recuperation resultat
  std::vector< unsigned char > h_resultat( W * H * 3);
  cudaMemcpy( h_resultat.data(), d_resultat, W * H * 3, cudaMemcpyDeviceToHost );

  //Sortie
  cv::Mat m_out( W, H, CV_8UC3, h_resultat.data() );
  cv::imwrite("out.jpg", m_out);

  //Erreur
  auto err = cudaGetLastError();
  if( err != cudaSuccess )
  {
    std::cout << cudaGetErrorString( err ) << "\n";
  }

  //Free
  cudaFree(d_resultat);
  cudaFree(d_filter);

  return 0;
}
