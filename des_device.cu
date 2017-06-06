#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "des.h"

__global__ void des_device (BYTE *data, BYTE *encrypted_data, 
    BYTE *decrypted_data, int st.st_size) {

    int idx = (threadIdx.x + blockDim.x * blockIdx.x) * DES_BLOCK_SIZE;
    int j, k;

    BYTE data_buf[DES_BLOCK_SIZE];
    BYTE data_enc[DES_BLOCK_SIZE];
    BYTE data_dec[DES_BLOCK_SIZE];

    BYTE key1[DES_BLOCK_SIZE] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF};
    BYTE schedule[16][6];

    for(j = 0; j < DES_BLOCK_SIZE; j++){
        if( (idx+j) < st.st_size){
            data_buf[j] = data[idx+j];
        };
    };

    des_key_setup(key1, schedule, DES_ENCRYPT);
    des_crypt(data_buf, data_enc, schedule);

    des_key_setup(key1, schedule, DES_DECRYPT);
    des_crypt(data_enc, data_dec, schedule);

    for(k = 0; k < DES_BLOCK_SIZE; k++){
        if((idx+k) < st.st_size){
            encrypted_data[idx+k] = data_enc[k];
            decrypted_data[idx+k] = data_dec[k];
        };
    };
}

void des_device_test_file(char* filename, int nblocks, int nthreads)
{
    BYTE *data, *encrypted_data, *decrypted_data;
    BYTE *d_data, *d_encrypted_data, *d_decrypted_data;
    int pass = 1;

    struct stat st;

    if (stat(filename, &st) == 0){
        data = (BYTE *) malloc(sizeof(BYTE) * st.st_size);
    };

    FILE *file = fopen(filename, "rb");

    if(data != NULL && file){
        int current_byte = 0;
        char filename_copy[80];
        char ext[5];

        while(fread(&data[current_byte], sizeof(BYTE), 1, file) == 1){
            current_byte += 1;
        };

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

    des_device <<<nblocks, nthreads>>>(d_data, d_encrypted_data, d_decrypted_data, st.st_size);

    cudaMemcpy(d_encrypted_data, encrypted_data, st.st_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_decrypted_data, decrypted_data, st.st_size, cudaMemcpyDeviceToHost);

    FILE *enc_file = fopen(filename, "wb+");
    FILE *dec_file = fopen(filename_copy, "wb+");

    fwrite(encrypted_data, sizeof(BYTE) * st.st_size, 1, enc_file);
    fwrite(decrypted_data, sizeof(BYTE) * st.st_size, 1, dec_file);

    fclose(enc_file);
    fclose(dec_file);

    cudaFree(d_data); cudaFree(d_encrypted_data); cudaFree(d_decrypted_data);
    free(data); free(encrypted_data); free(decrypted_data);
};

void des_device_test_all_files() {
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
    printf("DES DEVICE test file: %s ==> %s\n", filenames[i], 
      des_device_test_file(filenames[i], 4, 16) ? "SUCCEEDED" : "FAILED");
  }

}

int main (int argc, char** argv)
{
    if (argc != 3) {
        printf("Usage: ./des_device #blocks/grid  #threads/block\n");
        return -1;
    }

    int nblocks = atoi(argv[1]);
    int nthreads = atoi(argv[2]);

    printf("DES device test step 1: %s\n", des_device_test_file("sample_files/ulysses.txt", nblocks, nthreads) ? "SUCCEEDED" : "FAILED");
    //des_device_test_all_files();

    return 0;
}