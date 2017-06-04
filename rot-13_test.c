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
int rot13_test1()
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

int rot13_test2(char* filename)
{
    BYTE *data, *dataAux, *buf;
    int pass = 1;
    int n = strlen(filename);
    int i;

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

int main(int argc, char** argv)
{
    char filename[80];
    strcpy(filename, argv[1]);

    printf("ROT-13 test1: %s\n", rot13_test1() ? "SUCCEEDED" : "FAILED");
    printf("ROT-13 test2: %s\n", rot13_test2(filename) ? "SUCCEEDED" : "FAILED");

    return 0;
}
