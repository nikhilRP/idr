all: IDR_parameter_estimation.so

IDR_parameter_estimation.so:
	gcc -shared -fPIC -g -O0 IDR_parameter_estimation.c -lm -o IDR_parameter_estimation.so

clean:
	rm IDR_parameter_estimation.so
