/*********************************************************************
* Filename:   blowfish_test.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding Blowfish
              implementation. These tests do not encompass the full
              range of available test vectors, however, if the tests
              pass it is very, very likely that the code is correct
              and was compiled properly. This code also serves as
              example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <memory.h>
#include <sys/stat.h>
#include <stdlib.h>
#include "blowfish.h"

/*********************** FUNCTION DEFINITIONS ***********************/
int blowfish_test()
{
    BYTE key1[8]  = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    BYTE key2[8]  = {0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff};
    BYTE key3[24] = {0xF0,0xE1,0xD2,0xC3,0xB4,0xA5,0x96,0x87,
                     0x78,0x69,0x5A,0x4B,0x3C,0x2D,0x1E,0x0F,
                     0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77};
    BYTE p1[BLOWFISH_BLOCK_SIZE] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    BYTE p2[BLOWFISH_BLOCK_SIZE] = {0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff};
    BYTE p3[BLOWFISH_BLOCK_SIZE] = {0xFE,0xDC,0xBA,0x98,0x76,0x54,0x32,0x10};

    BYTE c1[BLOWFISH_BLOCK_SIZE] = {0x4e,0xf9,0x97,0x45,0x61,0x98,0xdd,0x78};
    BYTE c2[BLOWFISH_BLOCK_SIZE] = {0x51,0x86,0x6f,0xd5,0xb8,0x5e,0xcb,0x8a};
    BYTE c3[BLOWFISH_BLOCK_SIZE] = {0x05,0x04,0x4b,0x62,0xfa,0x52,0xd0,0x80};

    BYTE enc_buf[BLOWFISH_BLOCK_SIZE];
    BLOWFISH_KEY key;
    int pass = 1;

    // Test vector 1.
    blowfish_key_setup(key1, &key, BLOWFISH_BLOCK_SIZE);
    blowfish_encrypt(p1, enc_buf, &key);
    pass = pass && !memcmp(c1, enc_buf, BLOWFISH_BLOCK_SIZE);
    blowfish_decrypt(c1, enc_buf, &key);
    pass = pass && !memcmp(p1, enc_buf, BLOWFISH_BLOCK_SIZE);

    // Test vector 2.
    blowfish_key_setup(key2, &key, BLOWFISH_BLOCK_SIZE);
    blowfish_encrypt(p2, enc_buf, &key);
    pass = pass && !memcmp(c2, enc_buf, BLOWFISH_BLOCK_SIZE);
    blowfish_decrypt(c2, enc_buf, &key);
    pass = pass && !memcmp(p2, enc_buf, BLOWFISH_BLOCK_SIZE);

    // Test vector 3.
    blowfish_key_setup(key3, &key, 24);
    blowfish_encrypt(p3, enc_buf, &key);
    pass = pass && !memcmp(c3, enc_buf, BLOWFISH_BLOCK_SIZE);
    blowfish_decrypt(c3, enc_buf, &key);
    pass = pass && !memcmp(p3, enc_buf, BLOWFISH_BLOCK_SIZE);

    return(pass);
}

int blowfish_test_file(char* filename)
{
    BYTE *data, *encrypted_data, *decrypted_data;
    int i, j, k;
    int pass = 1;
    int n = strlen(filename);
    char filename_copy[80];
    
    BYTE key1[24] = {0xF0,0xE1,0xD2,0xC3,0xB4,0xA5,0x96,0x87,
                     0x78,0x69,0x5A,0x4B,0x3C,0x2D,0x1E,0x0F,
                     0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77};

    BYTE schedule[16][6];
    BLOWFISH_KEY key;
    BYTE enc_buf[BLOWFISH_BLOCK_SIZE];

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

    BYTE data_buf[BLOWFISH_BLOCK_SIZE];
    BYTE data_enc[BLOWFISH_BLOCK_SIZE];
    BYTE data_dec[BLOWFISH_BLOCK_SIZE];

    for (i = 0; i < st.st_size; i++) {
        for (j = 0; j < BLOWFISH_BLOCK_SIZE; j++) {
            if (i < st.st_size){
                data_buf[j] = data[i];
                i++;
            };
        };

	blowfish_key_setup(key1, &key, st.st_size);
	blowfish_encrypt(data_buf, data_enc, &key);
	blowfish_decrypt(data_enc, data_dec, &key);
	
	
        i -= BLOWFISH_BLOCK_SIZE;
        for (k = 0; k < BLOWFISH_BLOCK_SIZE; k++){
            if (i < st.st_size) {
                encrypted_data[i] = data_enc[k];
                decrypted_data[i] = data_dec[k];
                i++;
            };
        };

        i--;

        pass = pass && !memcmp(data_buf, data_dec, BLOWFISH_BLOCK_SIZE);
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


void blowfish_test_all_files() {
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
    printf("blowfish test file: %s ==> %s\n", filenames[i], 
      blowfish_test_file(filenames[i]) ? "SUCCEEDED" : "FAILED");
  }

}


int main(int argc, char** argv)
{
    if (argc != 2) {
        printf("Usage: ./blowfish_test <filename>\n");
        return -1;
    }

    printf("Blowfish tests 1: %s\n\n", blowfish_test_file(argv[1]) ? "SUCCEEDED" : "FAILED");
    // printf("Blowfish tests 2:\n");
    // blowfish_test_all_files();

    return(0);
}
