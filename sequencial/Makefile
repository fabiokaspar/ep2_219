ROT13=rot-13
DES=des
BLOWFISH=blowfish
AES=aes

CC=gcc
CC_OPT=

.PHONY: all
all: $(ROT13) $(DES) $(BLOWFISH) $(AES)

$(ROT13): $(ROT13)_test.c $(ROT13).c $(ROT13).h
	$(CC) -o $(ROT13) $(CC_OPT) $(ROT13)_test.c $(ROT13).c $(ROT13).h

$(DES): $(DES)_test.c $(DES).c $(DES).h
	$(CC) -o $(DES) $(CC_OPT1) $(DES)_test.c $(DES).c $(DES).h

$(BLOWFISH): $(BLOWFISH)_test.c $(BLOWFISH).c $(BLOWFISH).h
	$(CC) -o $(BLOWFISH) $(CC_OPT2) $(BLOWFISH)_test.c $(BLOWFISH).c $(BLOWFISH).h

$(AES): $(AES)_test.c $(AES).c $(AES).h
	$(CC) -o $(AES) $(CC_OPT2) $(AES)_test.c $(AES).c $(AES).h

.PHONY: clean
clean:
	rm $(ROT13) $(DES) $(BLOWFISH) $(AES) ../sample_files/*_enc* ../sample_files/*_dec*
