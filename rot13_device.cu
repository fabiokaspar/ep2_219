/*********************************************************************
* Filename:   rot-13.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Implementation of the ROT-13 encryption algorithm.
                  Algorithm specification can be found here:
                   *
                  This implementation uses little endian byte order.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "rot-13.h"

/*********************** FUNCTION DEFINITIONS ***********************/
__global__ void rot13(BYTE* str, int len)
{
   int case_type, idx;
   idx = blockIdx.x * blockDim.x + threadIdx.x;

   if (idx < len) {
      // Only process alphabetic characters.
      if (!(str[idx] < 'A' || (str[idx] > 'Z' && str[idx] < 'a') || str[idx] > 'z')){

      // Determine if the char is upper or lower case.
      if (str[idx] >= 'a')
         case_type = 'a';
      else
         case_type = 'A';
      // Rotate the char's value, ensuring it doesn't accidentally "fall off" the end.
      str[idx] = (str[idx] + 13) % (case_type + 26);
      if (str[idx] < 26)
         str[idx] += case_type;
      }
   }
}

int main (int argc, char** argv)
{
    if (argc != 2) {
      printf("Formato: ./rot13_device <arq_entrada>\n");
      return 1;
    }

    BYTE *data, *buf;
    BYTE *d_data;
    int pass = 1;
    int i;
    char* filename = argv[1];

    int n = strlen(filename);

    cudaMalloc((void**) &d_data, sizeof(BYTE) * 10);
    data = (BYTE *) malloc(sizeof(BYTE) * 10);
    buf = (BYTE *) malloc(sizeof(BYTE) * 10);
    FILE *file = fopen(filename, "rb");

    if (data != NULL && file) {

        char fname_aux[80];
        char ext[5];

        for (i = 0; i < 4; i++) {
            ext[i] = filename[n-4+i];
        }
        ext[4] = '\0';

        filename[n-4] = '\0';
        strcpy(fname_aux, filename);

        strcat(filename, "_enc");
        strcat(filename, ext);
        
        strcat(fname_aux, "_dec");
        strcat(fname_aux, ext);
        

        FILE *enc_file = fopen(filename, "wb+");
        FILE *dec_file = fopen(fname_aux, "wb+");

        while ((n = fread(data, sizeof(BYTE), 10, file)) > 0) {
            memcpy(buf, data, n);

            cudaMemcpy(d_data, data, n, cudaMemcpyHostToDevice);
            rot13 <<<4,16>>>(d_data, n);
            cudaMemcpy(data, d_data, n, cudaMemcpyDeviceToHost);
            fwrite(data, sizeof(BYTE), n, enc_file);


            cudaMemcpy(d_data, data, n, cudaMemcpyHostToDevice);
            rot13 <<<4,16>>>(d_data, n);
            cudaMemcpy(data, d_data, n, cudaMemcpyDeviceToHost);
            fwrite(data, sizeof(BYTE), n, dec_file);


            pass = pass && !memcmp(buf, data, n);
        }

        fclose(enc_file);
        fclose(dec_file);
    }

    fclose(file);

    return pass;
}
