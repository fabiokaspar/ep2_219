/*************************** HEADER FILES ***************************/
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include "blowfish.h"

/*********************** KERNELS (CUDA) ***********************/
__global__ void blowfish_device (BYTE *data, BYTE *encrypted_data, 
    BYTE *decrypted_data, int len) {

    int idx = (threadIdx.x + blockDim.x * blockIdx.x) * BLOWFISH_BLOCK_SIZE;
    int j, k;

    BYTE key2[8]  = {0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff};
    BLOWFISH_KEY key;

    BYTE data_buf[BLOWFISH_BLOCK_SIZE];
    BYTE data_enc[BLOWFISH_BLOCK_SIZE];
    BYTE data_dec[BLOWFISH_BLOCK_SIZE];

    for(j = 0; j < BLOWFISH_BLOCK_SIZE; j++){
        if( (idx+j) < len){
            data_buf[j] = data[idx+j];
        };
    };

    blowfish_key_setup(key2, &key, BLOWFISH_BLOCK_SIZE);
    blowfish_encrypt(data_buf, data_enc, &key);

    blowfish_decrypt(data_enc, data_dec, &key);

    for(k = 0; k < BLOWFISH_BLOCK_SIZE; k++){
        if((idx+k) < len){
            encrypted_data[idx+k] = data_enc[k];
            decrypted_data[idx+k] = data_dec[k];
        };
    };
}

/*********************** TEST FUNCTIONS ***********************/
int blowfish_device_test_file(char* filename, int nblocks, int nthreads)
{
    BYTE *data, *encrypted_data, *decrypted_data;
    BYTE *d_data, *d_encrypted_data, *d_decrypted_data;
    int i, pass = 1;
    int n = strlen(filename);
    char filename_copy[80];

    struct stat st;

    if (stat(filename, &st) == 0){
        data = (BYTE *) malloc(sizeof(BYTE) * st.st_size);
    };

    FILE *file = fopen(filename, "rb");

    if(data != NULL && file){
        int current_byte = 0;
        char ext[5];

        // le todo o arquivo e armazena no vetor data
        while(fread(&data[current_byte], sizeof(BYTE), 1, file) == 1){
            current_byte += 1;
        };

        // pega o nome do arquivo e acrescenta _enc ou _dec mais a extensao
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
    };

    encrypted_data = (BYTE *) malloc(sizeof(BYTE) * st.st_size);
    decrypted_data = (BYTE *) malloc(sizeof(BYTE) * st.st_size);

    cudaMalloc((void**) &d_data, sizeof(BYTE) * st.st_size);
    cudaMalloc((void**) &d_encrypted_data, sizeof(BYTE) * st.st_size);
    cudaMalloc((void**) &d_decrypted_data, sizeof(BYTE) * st.st_size);

    cudaMemcpy(d_data, data, st.st_size, cudaMemcpyHostToDevice);

    blowfish_device <<<nblocks, nthreads>>>(d_data, d_encrypted_data, d_decrypted_data, st.st_size);

    cudaMemcpy(d_encrypted_data, encrypted_data, st.st_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_decrypted_data, decrypted_data, st.st_size, cudaMemcpyDeviceToHost);

    pass = !memcmp(data, decrypted_data, st.st_size);
    
    FILE *enc_file = fopen(filename, "wb+");
    FILE *dec_file = fopen(filename_copy, "wb+");

    fwrite(encrypted_data, sizeof(BYTE) * st.st_size, 1, enc_file);
    fwrite(decrypted_data, sizeof(BYTE) * st.st_size, 1, dec_file);

    fclose(enc_file);
    fclose(dec_file);
    
    cudaFree(d_data); cudaFree(d_encrypted_data); cudaFree(d_decrypted_data);
    free(data); free(encrypted_data); free(decrypted_data);

    return pass;
};

void blowfish_device_test_all_files() {
  int i;
  char filenames[8][80] = 
      {"sample_files/hubble_1.tif", 
       "sample_files/hubble_2.png",
       "sample_files/hubble_3.tif",
       "sample_files/king_james_bible.txt",
       "sample_files/mercury.png",
       "sample_files/moby_dick.txt",
       "sample_files/tale_of_two_cities.txt",
       "sample_files/ulysses.txt"
  };

  for (i = 0; i < 8; i++) {
    printf("BLOWFISH DEVICE test file: %s ==> %s\n", filenames[i], 
      blowfish_device_test_file(filenames[i], 4, 16) ? "SUCCEEDED" : "FAILED");
  }

}

/*********************** MAIN FUNCTION ***********************/
int main (int argc, char** argv)
{
    if (argc != 4) {
        printf("Usage: ./blowfish_device #blocks/grid  #threads/block  <filename>\n");
        return -1;
    }

    int nblocks = atoi(argv[1]);
    int nthreads = atoi(argv[2]);

    printf("BLOWFISH device test step 1: %s\n", blowfish_device_test_file(argv[3], nblocks, nthreads) ? "SUCCEEDED" : "FAILED");
    //blowfish_device_test_all_files();

    return 0;
}
