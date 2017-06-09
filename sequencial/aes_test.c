/*********************************************************************
* Filename:   aes_test.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding AES
              implementation. These tests do not encompass the full
              range of available test vectors and are not sufficient
              for FIPS-140 certification. However, if the tests pass
              it is very, very likely that the code is correct and was
              compiled properly. This code also serves as
              example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "aes.h"

/*********************** FUNCTION DEFINITIONS ***********************/
int aes_ecb_test()
{
    WORD key_schedule[60], idx;
    BYTE enc_buf[128];
    BYTE plaintext[2][16] = {
        {0x6b,0xc1,0xbe,0xe2,0x2e,0x40,0x9f,0x96,0xe9,0x3d,0x7e,0x11,0x73,0x93,0x17,0x2a},
        {0xae,0x2d,0x8a,0x57,0x1e,0x03,0xac,0x9c,0x9e,0xb7,0x6f,0xac,0x45,0xaf,0x8e,0x51}
    };
    BYTE ciphertext[2][16] = {
        {0xf3,0xee,0xd1,0xbd,0xb5,0xd2,0xa0,0x3c,0x06,0x4b,0x5a,0x7e,0x3d,0xb1,0x81,0xf8},
        {0x59,0x1c,0xcb,0x10,0xd4,0x10,0xed,0x26,0xdc,0x5b,0xa7,0x4a,0x31,0x36,0x28,0x70}
    };
    BYTE key[1][32] = {
        {0x60,0x3d,0xeb,0x10,0x15,0xca,0x71,0xbe,0x2b,0x73,0xae,0xf0,0x85,0x7d,0x77,0x81,0x1f,0x35,0x2c,0x07,0x3b,0x61,0x08,0xd7,0x2d,0x98,0x10,0xa3,0x09,0x14,0xdf,0xf4}
    };
    int pass = 1;

    // Raw ECB mode.
    //printf("* ECB mode:\n");
    aes_key_setup(key[0], key_schedule, 256);
    //printf(  "Key          : ");
    //print_hex(key[0], 32);

    for(idx = 0; idx < 2; idx++) {
        aes_encrypt(plaintext[idx], enc_buf, key_schedule, 256);
        //printf("\nPlaintext    : ");
        //print_hex(plaintext[idx], 16);
        //printf("\n-encrypted to: ");
        //print_hex(enc_buf, 16);
        pass = pass && !memcmp(enc_buf, ciphertext[idx], 16);

        aes_decrypt(ciphertext[idx], enc_buf, key_schedule, 256);
        //printf("\nCiphertext   : ");
        //print_hex(ciphertext[idx], 16);
        //printf("\n-decrypted to: ");
        //print_hex(enc_buf, 16);
        pass = pass && !memcmp(enc_buf, plaintext[idx], 16);

        //printf("\n\n");
    }

    return(pass);
}

/*********************** TEST FUNCTIONS ***********************/
int aes_test_file(char* filename)
{
    BYTE *data, *encrypted_data, *decrypted_data;
    int i, j, k;
    int pass = 1;
    int n = strlen(filename);
    char filename_copy[80];

    WORD key_schedule[60];
    BYTE key[1][32] = {
        {0x60,0x3d,0xeb,0x10,0x15,0xca,0x71,0xbe,0x2b,0x73,0xae,0xf0,0x85,0x7d,0x77,0x81,0x1f,0x35,0x2c,0x07,0x3b,0x61,0x08,0xd7,0x2d,0x98,0x10,0xa3,0x09,0x14,0xdf,0xf4}
    };

    struct stat st;

    if (stat(filename, &st) == 0) {
        data = (BYTE *) malloc(sizeof(BYTE) * st.st_size);
    };

    FILE *file = fopen(filename, "rb");

    if (data != NULL && file) {
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

    BYTE data_buf[AES_BLOCK_SIZE];
    BYTE data_enc[AES_BLOCK_SIZE];
    BYTE data_dec[AES_BLOCK_SIZE];

    aes_key_setup(key[0], key_schedule, 256);

    for (i = 0; i < st.st_size; i++) {
        for (j = 0; j < AES_BLOCK_SIZE; j++) {
            if (i < st.st_size){
                data_buf[j] = data[i];
                i++;
            };
        };

        aes_encrypt(data_buf, data_enc, key_schedule, 256);
        aes_decrypt(data_enc, data_dec, key_schedule, 256);

        i -= AES_BLOCK_SIZE;
        for (k = 0; k < AES_BLOCK_SIZE; k++){
            if (i < st.st_size) {
                encrypted_data[i] = data_enc[k];
                decrypted_data[i] = data_dec[k];
                i++;
            };
        };

        i--;

        pass = pass && !memcmp(data_buf, data_dec, AES_BLOCK_SIZE);
    };

    FILE *enc_file = fopen(filename, "wb+");
    FILE *dec_file = fopen(filename_copy, "wb+");

    fwrite(encrypted_data, sizeof(BYTE) * st.st_size, 1, enc_file);
    fwrite(decrypted_data, sizeof(BYTE) * st.st_size, 1, dec_file);

    fclose(enc_file);
    fclose(dec_file);

    free(data); 
    free(encrypted_data); 
    free(decrypted_data);

    return pass;
};

void aes_test_all_files() {
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
    printf("AES test file: %s ==> %s\n", filenames[i], 
      aes_test_file(filenames[i]) ? "SUCCEEDED" : "FAILED");
  }

}

int main(int argc, char** argv)
{
    if (argc != 2) {
        printf("Usage: ./aes_test <filename>\n");
        return -1;
    }

    printf("AES test 1: %s\n\n", aes_test_file(argv[1]) ? "SUCCEEDED" : "FAILED");
    // printf("AES test 2:\n");
    // aes_test_all_files();

    return(0);
}
