ROT13=rot-13
DES=des
BLOWFISH=blowfish

CC=nvcc
CC_OPT=--gpu-architecture=sm_32

.PHONY: all
all: $(ROT13) $(DES) $(BLOWFISH)

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


$(ROT13): $(ROT13)_device.cu
	$(CC) --gpu-architecture=sm_32 -o $(ROT13) $(ROT13)_device.cu

.PHONY: clean
clean:
	rm $(ROT13) $(DES) $(BLOWFISH) *.o sample_files/*_enc* sample_files/*_dec* 
