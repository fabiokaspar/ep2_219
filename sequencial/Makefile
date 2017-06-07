ROT13=rot-13
DES=des
BLOWFISH=blowfish
ARCFOUR=arcfour

CC=gcc
CC_OPT=

.PHONY: all
all: $(ROT13) $(DES) $(BLOWFISH) $(ARCFOUR)

$(ROT13): $(ROT13)_test.c $(ROT13).c $(ROT13).h
	$(CC) -o $(ROT13) $(CC_OPT) $(ROT13)_test.c $(ROT13).c $(ROT13).h

$(DES): $(DES)_test.c $(DES).c $(DES).h
	$(CC) -o $(DES) $(CC_OPT1) $(DES)_test.c $(DES).c $(DES).h

$(BLOWFISH): $(BLOWFISH)_test.c $(BLOWFISH).c $(BLOWFISH).h
	$(CC) -o $(BLOWFISH) $(CC_OPT2) $(BLOWFISH)_test.c $(BLOWFISH).c $(BLOWFISH).h

$(ARCFOUR): $(ARCFOUR)_test.c $(ARCFOUR).c $(ARCFOUR).h
	$(CC) -o $(ARCFOUR) $(CC_OPT2) $(ARCFOUR)_test.c $(ARCFOUR).c $(ARCFOUR).h

.PHONY: clean
clean:
	rm $(ROT13) $(DES) $(BLOWFISH) $(ARCFOUR) ../sample_files/*_enc* ../sample_files/*_dec*
