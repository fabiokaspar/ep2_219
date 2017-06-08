/*********************************************************************
* Filename:   arcfour_test.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding ARCFOUR
              implementation. These tests do not encompass the full
              range of available test vectors, however, if the tests
              pass it is very, very likely that the code is correct
              and was compiled properly. This code also serves as
              example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include "arcfour.h"

/*********************** FUNCTION DEFINITIONS ***********************/
__global__ void block_arcfour(BYTE *data, BYTE *encrypted_data, BYTE *buf, int TAM_BLOCK, int n)
{
	int i;
    int idx = (blockDim.x * blockIdx.x + threadIdx.x)) * TAM_BLOCK;
    BYTE *input = &data[idx];
    BYTE *output = &encrypted_data[idx];
    int leng = TAM_BLOCK;
    if ((idx + TAM_BLOCK) > n)
    	leng = n - idx;

   for (i = 0; i < leng; i++)
   	output[i] = input[i] ^ buf[i];
}

int rc4_device_test_file(char* filename, int nthreads, char* key)
{
    BYTE *data, *buf, *encrypted_data, *decrypted_data;
    BYTE *d_data, *d_buf, *d_encrypted_data, *d_decrypted_data;
    BYTE state[256];
    int pass = 1;
    int n = strlen(filename);
    int i;
    int TAM_BLOCK = 1024;

    int threadsPerBlock = nthreads;
    int blocksPerGrid;

    struct stat st;
    cudaError err;

    if (stat(filename, &st) == 0){
      data = (BYTE *) malloc(sizeof(BYTE) * st.st_size);
      err = cudaMalloc((void**) &d_data, sizeof(BYTE) * st.st_size);
    };

    buf = (BYTE *) malloc(sizeof(BYTE) * TAM_BLOCK);
    err = cudaMalloc((void**) &d_buf, sizeof(BYTE) * TAM_BLOCK);
    FILE *file = fopen(filename, "rb");
    
    if (data != NULL && file) {
        int current_byte = 0;
        char filename_enc[80], filename_dec[80];
      
        strncpy(filename_enc, filename, n-4);
        filename_enc[n-4] = '\0';
        strcpy(filename_dec, filename_enc);        
        strcat(filename_enc, "_enc");
        strcat(filename_dec, "_dec");

        // grava a extensao em ext
        for (i = 0; i < 5; i++) {
            filename_enc[n + i] = filename[n-4+i];
            filename_dec[n + i] = filename[n-4+i];
        }
 
        FILE *enc_file = fopen(filename_enc, "wb+");
        FILE *dec_file = fopen(filename_dec, "wb+");

        while(fread(&data[current_byte], sizeof(BYTE), 1, file) == 1){
          current_byte += 1;
        };
        n = current_byte;
        blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        err = cudaMemcpy(d_data, data, n, cudaMemcpyHostToDevice);

        encrypted_data = (BYTE *) malloc(sizeof(BYTE) * st.st_size);
        decrypted_data = (BYTE *) malloc(sizeof(BYTE) * st.st_size);

        err = cudaMalloc((void**) &d_encrypted_data, sizeof(BYTE) * st.st_size);
        err = cudaMalloc((void**) &d_decrypted_data, sizeof(BYTE) * st.st_size);

        arcfour_key_setup(state, (BYTE *)key, strlen(key));    
        arcfour_generate_stream(state, buf, TAM_BLOCK);

        err = cudaMemcpy(d_buf, buf, TAM_BLOCK, cudaMemcpyHostToDevice);
        
        /*for (i = 0; i < n; i += TAM_BLOCK) {
          int leng = TAM_BLOCK;
          if ((i + TAM_BLOCK) > n)
            leng = n - i;
          block_arcfour(&data[i], &encrypted_data[i], buf, leng);
        }*/

        block_arcfour<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_encrypted_data, d_buf, TAM_BLOCK, n);
        err = cudaGetLastError();
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
          printf("Failed (error code %s)!\n", cudaGetErrorString(err));
          exit(EXIT_FAILURE);
        }
        
        err = cudaMemcpy(encrypted_data, d_encrypted_data, n, cudaMemcpyDeviceToHost);
        fwrite(encrypted_data, sizeof(BYTE), n, enc_file);
        
        block_arcfour<<<blocksPerGrid, threadsPerBlock>>>(d_encrypted_data, d_decrypted_data, d_buf, TAM_BLOCK, n);
        err = cudaGetLastError();
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
          printf("Failed (error code %s)!\n", cudaGetErrorString(err));
          exit(EXIT_FAILURE);
        }
        err = cudaMemcpy(decrypted_data, d_decrypted_data, n, cudaMemcpyDeviceToHost);
        fwrite(decrypted_data, sizeof(BYTE), n, dec_file);

        err = cudaFree(d_data);
        err = cudaFree(d_buf);
        err = cudaFree(d_encrypted_data);
        err = cudaFree(d_encrypted_data);

        pass = pass && !memcmp(decrypted_data, data, n);


        fclose(enc_file);
        fclose(dec_file);
    }

    fclose(file);

    return pass;
}



int rc4_test()
{
    BYTE state[256];
    BYTE key[3][10] = {{"Key"}, {"Wiki"}, {"Secret"}};
    BYTE stream[3][10] = {{0xEB,0x9F,0x77,0x81,0xB7,0x34,0xCA,0x72,0xA7,0x19},
                          {0x60,0x44,0xdb,0x6d,0x41,0xb7},
                          {0x04,0xd4,0x6b,0x05,0x3c,0xa8,0x7b,0x59}};
    int stream_len[3] = {10,6,8};
    BYTE buf[1024];
    int idx;
    int pass = 1;

    // Only test the output stream. Note that the state can be reused.
    for (idx = 0; idx < 3; idx++) {
        arcfour_key_setup(state, (BYTE *)key[idx], strlen(key[idx]));
        arcfour_generate_stream(state, buf, stream_len[idx]);
        pass = pass && !memcmp(stream[idx], buf, stream_len[idx]);
    }

    return(pass);
}

void arcfour_device_test_all_files(int nthreads) {
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

  for (i = 0; i < 8; i++)
    printf("ARCFOUR DEVICE test file: %s ==> %s\n", filenames[i], rc4_device_test_file(filenames[i], nthreads, "Secret") ? "SUCCEEDED" : "FAILED");
}

/*int main()
{
    printf("ARCFOUR tests: %s\n", rc4_test_file("sample_files/hubble_1.tif", "Secret") ? "SUCCEEDED" : "FAILED");

    return(0);
}*/

int main (int argc, char** argv)
{
    if (argc != 3) {
        printf("Usage: ./arcfour_device #threads/block\n");
        return -1;
    }

    int nthreads = atoi(argv[1]);

    arcfour_device_test_all_files(nthreads);
    

    return 0;
}