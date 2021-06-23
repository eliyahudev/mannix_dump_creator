#ifndef MANNIX_ACCELERATOR
#define MANNIX_ACCELERATOR


typedef struct Matrix_vol32 {
    int rows;
    int cols;
    int size;
    int pad_size;
    void* data;    
}Matrix_vol32;

typedef struct Tensor_vol32 {
    int rows;
    int cols;
    int depth;
    Matrix_vol32* matrix;  
}Tensor_vol32;

typedef struct Tensor4D_vol32 {
    int rows;
    int cols;
    int depth;
    int dim;
    Tensor_vol32* tensor;
} Tensor4D_vol32;

// -------- allocators ----------

typedef struct Allocator_vol32 {
    int index;
    int max_size;
    void* data;
} Allocator_vol32 ;

typedef struct MatAllocator_vol32 {
    int index;
    int max_size;
    Matrix_vol32 * matrix;
} MatAllocator_vol32 ;

typedef struct TensorAllocator_vol32 {
    int index;
    int max_size;
    Tensor_vol32* tensor;
} TensorAllocator_vol32 ;

// ================== matrix functions ========================
// ================= int allocation ============================
Allocator_vol32* createAllocator_vol32(Allocator_vol32* alloc, void * data, int max_size) {
    alloc[0].index = 0;
    alloc[0].max_size = max_size-1;
    alloc[0].data = data;
    return alloc;
}

MatAllocator_vol32  * createMatrixAllocator_vol32 ( MatAllocator_vol32 * mat_alloc, Matrix_vol32* matrix, int max_size) {
    mat_alloc->index = 0;
    mat_alloc->max_size = max_size-1;
    mat_alloc->matrix = matrix;
    return mat_alloc;
}

TensorAllocator_vol32* createTensorAllocator_vol32(TensorAllocator_vol32* tens_alloc, Tensor_vol32* tens, int max_size) {
    tens_alloc->index = 0;
    tens_alloc->max_size = max_size-1;
    tens_alloc->tensor = tens;
    return tens_alloc;
}

void* mannixDataMalloc_vol32(Allocator_vol32* alloc, int length) {
        
    int byte_length = length*4 ;
    if (alloc[0].index + byte_length < alloc[0].max_size)
        alloc->index += byte_length;
    else {
        printf("ERROR-out of range allocation");
        exit(-1);
    }
    return alloc->data + alloc->index - byte_length; 
}

Matrix_vol32* mannixMatrixMalloc_vol32(MatAllocator_vol32* mat_alloc, int length) {
    if (mat_alloc->index + length < mat_alloc->max_size)  mat_alloc->index += length;
    else {
        printf("ERROR-out of range allocation");
        exit(-1);
    }
    return mat_alloc->matrix + mat_alloc->index -length; 
}

Tensor_vol32* mannixTensorMalloc_vol32(TensorAllocator_vol32* tens_alloc, int length) {
    if (tens_alloc->index + length < tens_alloc->max_size)
        tens_alloc->index += length;
    else {
        printf("ERROR-out of range allocation");
        exit(-1);
    }
    return tens_alloc->tensor + tens_alloc->index -length; 
}

    // ------------------matrix creation -------------------------
    // cearte matrix with the value of zero 
Matrix_vol32* creatMatrix_vol32(int rows, int cols, Matrix_vol32* m1, Allocator_vol32* al) {
        srand(time(NULL));
        m1[0].cols = cols;
        m1[0].rows = rows;
        m1->size = cols * rows;
        m1->pad_size = m1->size;
        m1[0].data = mannixDataMalloc_vol32(al, m1->pad_size);
        for (int i = 0; i < m1->pad_size; i++) {
             
            *((int*) m1->data + 4*i) = i;//rand() % 100 ;
         }

        return m1;
    }


Tensor_vol32* createTensor_vol32(int rows, int cols, int depth, Tensor_vol32* tens, Allocator_vol32* al, MatAllocator_vol32* mat_alloc) {

    tens->rows = rows;
    tens->cols = cols;
    tens->depth = depth;
    tens->matrix = mannixMatrixMalloc_vol32(mat_alloc, depth);
    for(int i = 0; i < depth; i++) {
        creatMatrix_vol32(rows, cols, &tens->matrix[i], al);
    }
    return  tens;
}


void create4DTensor_vol32(Tensor4D_vol32* tens_4d, int rows, int cols, int depth, int dim, Allocator_vol32* al, MatAllocator_vol32* mat_alloc, TensorAllocator_vol32* tens_alloc) {
    tens_4d->rows = rows;
    tens_4d->cols = cols;
    tens_4d->depth = depth;
    tens_4d->dim = dim;
    tens_4d->tensor = mannixTensorMalloc_vol32(tens_alloc, dim);

    for(int i = 0; i < tens_4d->dim; i++)
        createTensor_vol32(rows, cols, depth, &tens_4d->tensor[i], al, mat_alloc);
}


void printMatrix_vol32(Matrix_vol32* m1) {
    if(m1->rows <= 0 || m1->cols <= 0) {
        printf("Dimension ERRER - non positive hight or width  [%d][%d]\n",m1-> cols, m1->rows);
        exit(-1);
    }
    int i=0;
    while(i<m1[0].rows * m1[0].cols) {
        if (i % m1[0].cols == 0)
            printf(" [");
        printf("%d ", *((int*)m1->data + (i++)*4));
        if (i % m1[0].cols == 0 && i<m1[0].rows * m1[0].cols)
            printf("]\n");
        else if (i % m1[0].cols == 0)
            printf("]");
    }
    // printf("\n row size = %d, col size = %d\n\n", m1[0].rows, m1[0].cols);
}

void printTensor_vol32(Tensor_vol32* tens) {

    for(int i = 0; i < tens->depth; i++) {
        printMatrix_vol32(&tens->matrix[i]);
        if(i!= tens->depth-1) 
            printf("\n,\n");        
    }
}

void print4DTensor_vol32(Tensor4D_vol32* tens_4d) {

    printf("tensor{\n");    
    for(int i = 0; i < tens_4d->dim; i++) {
        printf("\n[");
        printTensor_vol32(&tens_4d->tensor[i]);
        printf(" ]\n");
    }
    printf("}\n");
    printf("rows size: %d columns size: %d depth size: %d dim: %d \n\n",tens_4d->rows,tens_4d->cols,tens_4d->depth,tens_4d->dim);

}

// Mannif Format (mfdb , mfds) Access functions

void dump_model_params_mfdb_vol32(Allocator_uint8 *alloc, char *dumpFileName) {
    
    FILE* dumpFile = fopen(dumpFileName,"w");

    printf("Dumping model parameters to %s\n",dumpFileName); 
    for (int i=0; i<alloc->index; i = i+4) {
        int word32 = 0x00000000; 
        word32 = word32 | alloc->data[i];
        word32 = word32 | (int)(alloc->data[i+1]) << 8*1;
        word32 = word32 | (int)(alloc->data[i+2]) << 8*2;
        word32 = word32 | (int)(alloc->data[i+3]) << 8*3;
        printf("-----------\n");
        printf("%08x - i=%d\n",alloc->data[i],i);
        printf("%08x - i=%d\n",alloc->data[i+1],i+1);
        printf("%08x - i=%d\n",alloc->data[i+2],i+2);
        printf("%08x - i=%d\n",alloc->data[i+3],i+3);
        printf("-----------\n");
        printf("%08x - i=%d\n",word32,i);
        printf("-----------\n");
        fprintf(dumpFile,"%08x\n",word32);
    }
        
    fclose(dumpFile) ;
    printf("\n\nDONE Dumping vol32\n\n");
    // exit(0);
}
#endif