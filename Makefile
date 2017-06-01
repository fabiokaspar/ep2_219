OUTPUT=rot-13

CC=gcc
CC_OPT=

.PHONY: all
all: $(OUTPUT)

$(OUTPUT): $(OUTPUT)_test.c $(OUTPUT).c $(OUTPUT).h
	$(CC) -o $(OUTPUT) $(CC_OPT) $(OUTPUT)_test.c $(OUTPUT).c $(OUTPUT).h

.PHONY: clean
clean:
	rm $(OUTPUT) sample_files/*_enc* sample_files/*_dec*
