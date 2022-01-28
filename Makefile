CC:=nvcc
EXE:=brandes
OBJS:=src/main.o
CFLAGS:=-O2 -gencode arch=compute_70,code=sm_70
INCLUDE:=-I./include

%.o: %.cu
	$(CC) $(CFLAGS) $(INCLUDE) $< -c -o $@

$(EXE): $(OBJS)
	$(CC) $^ -o $@

clean:
	rm -f $(OBJS) $(EXE)