DES=des
BLOWFISH=blowfish
AES=aes

CC=nvcc
CC_OPT=--gpu-architecture=sm_32

.PHONY: all
all: $(ROT13) $(DES) $(BLOWFISH) $(AES)


$(BLOWFISH): $(BLOWFISH)_device.o $(BLOWFISH).o
	$(CC) --gpu-architecture=sm_32 -o $(BLOWFISH) $(BLOWFISH)_device.o $(BLOWFISH).o

$(BLOWFISH)_device.o: $(BLOWFISH)_device.cu
	$(CC) --gpu-architecture=sm_32 --device-c $(BLOWFISH)_device.cu

$(BLOWFISH).o: $(BLOWFISH).cu
	$(CC) --gpu-architecture=sm_32 --device-c $(BLOWFISH).cu



$(DES): $(DES)_device.o $(DES).o
	$(CC) --gpu-architecture=sm_32 -o $(DES) $(DES)_device.o $(DES).o

$(DES)_device.o: $(DES)_device.cu
	$(CC) --gpu-architecture=sm_32 --device-c $(DES)_device.cu

$(DES).o: $(DES).cu
	$(CC) --gpu-architecture=sm_32 --device-c $(DES).cu


$(AES): $(AES)_device.o $(AES).o
	$(CC) --gpu-architecture=sm_32 -o $(AES) $(AES)_device.o $(AES).o

$(AES)_device.o: $(AES)_device.cu
	$(CC) --gpu-architecture=sm_32 --device-c $(AES)_device.cu

$(AES).o: $(AES).cu
	$(CC) --gpu-architecture=sm_32 --device-c $(AES).cu



.PHONY: clean
clean:
	rm $(ROT13) $(DES) $(BLOWFISH) $(AES) *.o ../sample_files/*_enc* ../sample_files/*_dec* 
