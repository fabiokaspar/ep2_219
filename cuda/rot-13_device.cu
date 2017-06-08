/*************************** HEADER FILES ***************************/
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "rot-13.h"

/*********************** KERNELS (CUDA) ***********************/
__global__ void rot13_device(BYTE* str, int len)
{
   int case_type, idx;
   idx = blockIdx.x * blockDim.x + threadIdx.x;

   if (idx < len) {
      // Only process alphabetic characters.
      if (!(str[idx] < 'A' || (str[idx] > 'Z' && str[idx] < 'a') || str[idx] > 'z')) {
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

/*********************** TEST FUNCTIONS ***********************/
int rot13_device_test(int nblocks, int nthreads)
{
    BYTE text[] = {"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"};
    BYTE code[] = {"NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm"};
    BYTE buf[52];
    BYTE d_buf[52];
    int pass = 1;

    // To encode, just apply ROT-13.
    memcpy(buf, text, 52);

    cudaMalloc((void**) &d_buf, sizeof(BYTE) * 52);
    cudaMemcpy(d_buf, buf, 52, cudaMemcpyHostToDevice);
    rot13_device <<<nblocks, nthreads>>>(d_buf, 52);

    cudaMemcpy(buf, d_buf, 52, cudaMemcpyDeviceToHost);

    pass = pass && !memcmp(code, buf, 52);

    // To decode, just re-apply ROT-13.
    rot13_device <<<nblocks, nthreads>>>(d_buf, 52);
    
    cudaMemcpy(buf, d_buf, 52, cudaMemcpyDeviceToHost);
    
    pass = pass && !memcmp(text, buf, 52);
    cudaFree(d_buf);

    return pass;
}

int rot13_device_test_file(char* filename, int nblocks, int nthreads) {
    BYTE *data, *buf;
    BYTE *d_data;
    int pass = 1;
    int i;

    int n = strlen(filename);

    cudaMalloc((void**) &d_data, sizeof(BYTE) * 100);
    data = (BYTE *) malloc(sizeof(BYTE) * 100);
    buf = (BYTE *) malloc(sizeof(BYTE) * 100);

    FILE *file = fopen(filename, "rb");

    if (data && file && buf) {

        char filename_copy[80];
        char ext[5];

        for (i = 0; i < 4; i++) {
            ext[i] = filename[n-4+i];
        }
        ext[4] = '\0';

        filename[n-4] = '\0';
        strcpy(filename_copy, filename);

        strcat(filename, "_enc");
        strcat(filename, ext);
        
        strcat(filename_copy, "_dec");
        strcat(filename_copy, ext);
        

        FILE *enc_file = fopen(filename, "wb+");
        FILE *dec_file = fopen(filename_copy, "wb+");

        while ((n = fread(data, sizeof(BYTE), 100, file)) > 0) {
            memcpy(buf, data, n);

            cudaMemcpy(d_data, data, n, cudaMemcpyHostToDevice);
            rot13_device <<<nblocks, nthreads>>>(d_data, n);
            
            cudaMemcpy(data, d_data, n, cudaMemcpyDeviceToHost);
            fwrite(data, sizeof(BYTE), n, enc_file);

            //cudaMemcpy(d_data, data, n, cudaMemcpyHostToDevice);
            rot13_device <<<nblocks, nthreads>>>(d_data, n);
            
            cudaMemcpy(data, d_data, n, cudaMemcpyDeviceToHost);
            fwrite(data, sizeof(BYTE), n, dec_file);

            pass = pass && !memcmp(buf, data, n);
        }

        fclose(enc_file);
        fclose(dec_file);
    }

    fclose(file);
    cudaFree(d_data);
    free(data);

    return pass;
}

void rot13_device_test_all_files() {
  int i;
  char filenames[8][80] = {
       "../sample_files/hubble_1.tif", 
       "../sample_files/hubble_2.png",
       "../sample_files/hubble_3.tif",
       "../sample_files/king_james_bible.txt",
       "../sample_files/mercury.png",
       "../sample_files/moby_dick.txt",
       "../sample_files/tale_of_two_cities.txt",
       "../sample_files/ulysses.txt"
  };

  for (i = 0; i < 8; i++) {
    printf("ROT-13 DEVICE test file: %s ==> %s\n", filenames[i], 
      rot13_device_test_file(filenames[i], 4, 16) ? "SUCCEEDED" : "FAILED");
  }

}

/*********************** MAIN FUNCTION ***********************/
int main (int argc, char** argv)
{
    if (argc != 4) {
        printf("Usage: ./rot13_device #blocks/grid  #threads/block <filename>\n");
        return -1;
    }

    int nblocks = atoi(argv[1]);
    int nthreads = atoi(argv[2]);

    printf("ROT-13 device test step 1: %s\n", rot13_device_test_file(argv[3], nblocks, nthreads) ? "SUCCEEDED" : "FAILED");
    //rot13_device_test_all_files();

    return 0;
}
