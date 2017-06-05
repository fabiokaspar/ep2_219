/*********************************************************************
* Filename:   rot-13_test.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding ROT-13
              implementation. These tests do not encompass the full
              range of available test vectors, however, if the tests
              pass it is very, very likely that the code is correct
              and was compiled properly. This code also serves as
              example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rot-13.h"

/*********************** FUNCTION DEFINITIONS ***********************/
int rot13_test()
{
    char text[] = {"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"};
    char code[] = {"NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm"};
    char buf[1024];
    int pass = 1;

    // To encode, just apply ROT-13.
    strcpy(buf, text);
    rot13(buf);
    pass = pass && !strcmp(code, buf);

    // To decode, just re-apply ROT-13.
    rot13(buf);
    pass = pass && !strcmp(text, buf);

    return(pass);
}

int rot13_test_file(char* filename)
{
    BYTE *data, *buf;
    int pass = 1;
    int n = strlen(filename);
    int i;

    data = (BYTE *) malloc(sizeof(BYTE) * 10);
    buf = (BYTE *) malloc(sizeof(BYTE) * 10);
    FILE *file = fopen(filename, "rb");

    if (data != NULL && file) {

        char filename_copy[80];
        char ext[5];

        // grava a extensao em ext
        for (i = 0; i < 4; i++) {
            ext[i] = filename[n-4+i];
        }
        ext[4] = '\0';

        // apaga a extensão
        filename[n-4] = '\0';

        strcpy(filename_copy, filename);

        // anexa o sufixo _enc."ext"
        strcat(filename, "_enc");
        strcat(filename, ext);
        
        strcat(filename_copy, "_dec");
        strcat(filename_copy, ext);
        

        FILE *enc_file = fopen(filename, "wb+");
        FILE *dec_file = fopen(filename_copy, "wb+");

        while ((n = fread(data, sizeof(BYTE), 10, file)) > 0) {
            memcpy(buf, data, n);

            rot13(data);
            fwrite(data, sizeof(BYTE), n, enc_file);

            
            rot13(data);
            fwrite(data, sizeof(BYTE), n, dec_file);

            pass = pass && !memcmp(buf, data, n);
        }

        fclose(enc_file);
        fclose(dec_file);
    }

    fclose(file);

    return pass;
}

void rot13_test_all_files() {
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
        printf("ROT-13 test file: %s ==> %s\n", filenames[i], rot13_test_file(filenames[i]) ? "SUCCEEDED" : "FAILED");   
    }
}

int main(int argc, char** argv)
{
    printf("ROT-13 test step 1: %s\n", rot13_test() ? "SUCCEEDED" : "FAILED");
    printf("ROT-13 test step 2:\n\n");
    rot13_test_all_files();

    return 0;
}