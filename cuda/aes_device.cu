/*************************** HEADER FILES ***************************/
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include "aes.h"

/*********************** KERNELS (CUDA) ***********************/
__global__ void aes_ecb_device (BYTE *data, BYTE *encrypted_data, 
    BYTE *decrypted_data, int len) {

    int idx = (threadIdx.x + blockDim.x * blockIdx.x) * AES_BLOCK_SIZE;
    int j, k;

    BYTE data_buf[AES_BLOCK_SIZE];
    BYTE data_enc[AES_BLOCK_SIZE];
    BYTE data_dec[AES_BLOCK_SIZE];

    WORD key_schedule[60];
    BYTE key[1][32] = {
        {0x60,0x3d,0xeb,0x10,0x15,0xca,0x71,0xbe,0x2b,0x73,0xae,0xf0,0x85,0x7d,0x77,0x81,0x1f,0x35,0x2c,0x07,0x3b,0x61,0x08,0xd7,0x2d,0x98,0x10,0xa3,0x09,0x14,0xdf,0xf4}
    };  

    for(j = 0; j < AES_BLOCK_SIZE; j++){
        if( (idx+j) < len){
            data_buf[j] = data[idx+j];
        };
    };

    aes_key_setup(key[0], key_schedule, 256);       //poderia ficar fora da gpu
                                                    //pois eh calculada uma vez apenas

    aes_encrypt(data_buf, data_enc, key_schedule, 256);
    aes_decrypt(data_enc, data_dec, key_schedule, 256);

    for(k = 0; k < AES_BLOCK_SIZE; k++){
        if((idx+k) < len){
            encrypted_data[idx+k] = data_enc[k];
            decrypted_data[idx+k] = data_dec[k];
        };
    };
}

/*********************** TEST FUNCTIONS ***********************/
int aes_device_test_file(char* filename, int threadsPerBlock)
{
    BYTE *data, *encrypted_data, *decrypted_data;
    BYTE *d_data, *d_encrypted_data, *d_decrypted_data;
    int pass = 1;
    int blocoDes, blocksPerGrid, i, n = strlen(filename);
    char filename_copy[80];

    struct stat st;

    if (stat(filename, &st) == 0){
        data = (BYTE *) malloc(sizeof(BYTE) * st.st_size);
    };

    blocoDes = (st.st_size + AES_BLOCK_SIZE - 1) / AES_BLOCK_SIZE;     
    blocksPerGrid = (blocoDes + threadsPerBlock - 1) / threadsPerBlock; 

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

    aes_ecb_device <<<blocksPerGrid, threadsPerBlock>>>(d_data, d_encrypted_data, d_decrypted_data, st.st_size);

    cudaMemcpy(encrypted_data, d_encrypted_data, st.st_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(decrypted_data, d_decrypted_data, st.st_size, cudaMemcpyDeviceToHost);

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

void aes_device_test_all_files() {
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
    printf("AES DEVICE test file: %s ==> %s\n", filenames[i], 
      aes_device_test_file(filenames[i], 16) ? "SUCCEEDED" : "FAILED");
  }

}

/*********************** MAIN FUNCTION ***********************/
int main (int argc, char** argv)
{
    if (argc != 3) {
        printf("Usage: ./aes_device #threads/block <filename>\n");
        return -1;
    }

    int nthreads = atoi(argv[1]);

    printf("AES device test step 1: %s\n", aes_device_test_file(argv[2], nthreads) ? "SUCCEEDED" : "FAILED");
    //aes_device_test_all_files();

    return 0;
}
