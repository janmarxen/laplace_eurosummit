/*
	Purpose: Created for EuroHPC Masters students
	Author: Gaurav Saxena
	Date: 17th Feb 2025
	Program: implements Jacobi iterative solver for Laplace Equation in 2-D by checkerboard decomposition. 
			 using a 2-D Cartesian topology of MPI processes. 
	Restriction: Do not enter a prime no. of processes e.g. 17, 19, 23, 47 etc. 
	For ease of study: Enter the same number of rows as the columns for the whole domain e.g. 501x501, 113x113
*/

#include<stdio.h>
#include<mpi.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

#define Tol 0.0001
#define SAVE_ITER 400
#define PROB_SIZE 257

////////////////////////////////////////MATRIX ALLOCATION/////////////////////////////////

float** allocate_mem(int rows, int columns)
{
	int i,j;
	float **matrix;
	float *arr;
	
	arr = (float*)calloc(rows*columns, sizeof(float));
	
	matrix = (float**)calloc(rows, sizeof(float*));
	
	for(i = 0; i<= rows-1; i++)
		matrix[i] = &(arr[i*columns]);	//Fills up matrix element with 0
		
	return matrix ; 
}

///////////////////////////////////////INDEPENDENT UPDATE/////////////////////////////////

float independent_update(float **matrix, float **new_matrix, int total_rows, int total_columns)
{
	
	int i, j ;
	float error = 0, diff;

	for(i = 2 ; i <= total_rows-3 ; i++)			//index of last row = total_rows-1, total_rows-2, total_rows-3
		for(j = 2 ; j <= total_columns-3; j++)
		{
			new_matrix[i][j] = 0.25*(matrix[i+1][j]+matrix[i-1][j]+matrix[i][j+1]+matrix[i][j-1]);
			diff = new_matrix[i][j] - matrix[i][j];
			diff = (diff > 0 ? diff : -1.0 * diff) ;
			if(diff > error)
				error = diff;
		}
	return error;
}

///////////////////////////////////////DEPENDENT ROW //////////////////////////////////////


float dependent_row_update(float **matrix, float **new_matrix, int row, int total_columns)
{

	int j ;
	float error = 0, diff;
	
	for(j = 1 ; j<= total_columns-2; j++)
	{
		new_matrix[row][j] = 0.25*(matrix[row+1][j]+matrix[row-1][j]+matrix[row][j+1]+matrix[row][j-1]);	
		diff = new_matrix[row][j] - matrix[row][j];
		diff = (diff > 0 ? diff : -1.0 * diff) ;
			if(diff > error)
				error = diff;
	}
	return error;
}

///////////////////////////////////////DEPENDENT COLUMN //////////////////////////////////////

float dependent_col_update(float **matrix, float **new_matrix, int col, int total_rows)
{

	int i ;
	float error = 0, diff;
	
	for(i = 1 ; i<= total_rows-2; i++)
	{
		new_matrix[i][col] = 0.25*(matrix[i+1][col]+matrix[i-1][col]+matrix[i][col+1]+matrix[i][col-1]);	
		diff = new_matrix[i][col] - matrix[i][col];
		diff = (diff > 0 ? diff : -1.0 * diff) ;
			if(diff > error)
				error = diff;
	}
	return error;
}

///////////////////////////////////////DISPLAY ///////////////////////////////////////////

void display(float **matrix, int rows, int cols)
{
	int i, j ;
	for(i = 0 ; i <= rows-1; i++)
	{
		for(j = 0 ; j<= cols-1; j++)
			printf(" %6.4f ",matrix[i][j]);
		printf("\n");
	}
}

void save_current_solution(float **matrix, int prows, int pcols, MPI_Comm new_comm, int N, int iterations)
{
	float *point_val;
	int rank, size;
	int coords[2];
	int counter = 0; 
	float h = 1.0/N ; 
	MPI_File fh ;
	MPI_Offset offset; 
	MPI_Aint extent;
	char file_name[30];
	int i,j ;
	float *bndry_vals ;   

	MPI_Comm_rank(new_comm, &rank); 
	MPI_Comm_size(new_comm, &size); 

	point_val = (float *) malloc(prows * pcols * 3 * sizeof(float*)); 
	
	MPI_Cart_coords(new_comm, rank, 2, coords);

	for(i = 1; i <= prows; i++)
	{
		for(j = 1; j<= pcols; j++)
		{
			point_val[counter++]= coords[1] * pcols * h + j * h; 	//global x-coordinate
			point_val[counter++]= coords[0] * prows * h + i * h; 	//global y-coordinate
			point_val[counter++]= matrix[i][j];  
		}
	}

	sprintf(file_name, "%d", iterations); 
	strcat(file_name,".dat"); 

	MPI_File_open(new_comm, file_name, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
	MPI_Type_extent(MPI_FLOAT, &extent);
	offset = rank * prows * pcols * 3 * extent; 	
	MPI_File_set_view(fh, offset, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
	MPI_File_write(fh, point_val, prows * pcols * 3, MPI_FLOAT, MPI_STATUS_IGNORE);
	
	// Note: We will not close the file here as we need to write boundary values below


	bndry_vals = (float *)malloc( (N + 1 + N + 1 + N - 1 + N - 1) * 3 * sizeof(float) ); //Allocate bndry_vals array
	offset = size * prows * pcols * 3 * extent;											 //Need to write AFTER all data written above
	MPI_File_set_view(fh, offset, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);		 //Set view though will need only on rank = 0

	//Write the boundary values using the master-rank ONLY
	//We will write N+1 values on the upper boundary AND the lower boundary
	//We will write only N - 1 values in the left boundary AND right boundary to avoid repetition

	if(rank == 0)
	{
	    counter = 0;

	    // Upper boundary (y = 0)
	    j = 0;
	    for(i = 0; i <= N ; i++) 
	    {
		double x = i * h;
		bndry_vals[counter++] = x;
		bndry_vals[counter++] = j * h;
		bndry_vals[counter++] = 1+sin(M_PI * x); // Use sine function
	    }

	    // Lower boundary (y = 1)
	    j = N;
	    for(i = 0; i <= N ; i++) 
	    {
		double x = i * h;
		bndry_vals[counter++] = x;
		bndry_vals[counter++] = j * h;
		bndry_vals[counter++] = 1+sin(M_PI * x); // Use sine function
	    }

	    // Left boundary (x = 0)
	    i = 0;
	    for(j = 1; j <= N-1 ; j++) 
	    {
		double y = j * h;
		bndry_vals[counter++] = i * h;
		bndry_vals[counter++] = y;
		bndry_vals[counter++] = 1+sin(M_PI * y); // Use sine function
	    }

	    // Right boundary (x = 1)
	    i = N;
	    for(j = 1; j <= N-1 ; j++) 
	    {
		double y = j * h;
		bndry_vals[counter++] = i * h;
		bndry_vals[counter++] = y;
		bndry_vals[counter++] = 1+sin(M_PI * y); // Use sine function
	    }

	    MPI_File_write(fh, bndry_vals, 4 * N * 3, MPI_FLOAT, MPI_STATUS_IGNORE);
	}

	MPI_File_close(&fh); 						//Finally close the file
}

int main()
{

///////////////////////////////////VARIABLE DECLARATION///////////////////////////////////

	int *N;							//Array to specify size in each dimension
	int ndims; 						//Whether 2-D or 3-D (will be useful later) 
	int P;							//Number of processes
	MPI_Comm old_comm, new_comm; 	//old_comm = MPI_COMM_WORLD, new_comm used in MPI_Cart_create
	int size, rank; 				//Number of processes in old_comm, rank of each process
	int *dims;						//Will contain size of each dimension in Cartesian Topology
	int i, j; 						//Loop variables
	int prows, pcols;				//Rows and columns of unknowns to each process
	float **old, **new, **temp;		//old and new matrices; 
	int reorder;					//Whether processes can be reordered in new topology or not
	int *period;					//Periodicity at boundaries, here it is period[2], in 3-D will be period[3]
	int up, down;					//Neighbour up and down
	int left, right;				//Neighbour left and right
	int displacement;				//Displacement for finding neighbours, Here it is = 1
	int direction;					//The dimension in which we find neighbours 0,1 for 2-D, 0,1,2 for 3-D
	MPI_Datatype row_type;			//Contiguous data for upper and lower halos
	MPI_Datatype col_type;			//Vector data for columns
	int count, blocklength, stride;	//For defining col_type; 
	MPI_Request recv[4];			//Receive request for Irecv()
	MPI_Request send[4]; 			//Send request for MPI_Isend();
	float G_max_err =1.0; 			//Global maximum error, program terminates when G_max_err < Tol 
	float local_max_err; 			//For finding maximum error on each process
	float local_max_err_new;		//For comparing with local_max_err 
	int iterations = 0; 			//Counting total iterations
	double start, end; 				//For measuring time of parallel program
	
////////////////////////////////////MPI INITIALISATION, SIZE, RANK CALLS//////////////////	

	MPI_Init(NULL, NULL);
	start = MPI_Wtime(); 			
	old_comm = MPI_COMM_WORLD; 		
	MPI_Comm_size(old_comm, &size);	
	MPI_Comm_rank(old_comm, &rank);	
	P = size; 						//P = number of processes in old_comm i.e. MPI_COMM_WORLD
	

///////////////////////////////////BROADCAST DIMENSIONS, DEFINE N[2], DEFINE dims[2]//////
		
		ndims = 2;
		MPI_Bcast(&ndims, 1, MPI_INTEGER, 0, old_comm);
		N = 		(int *)malloc(sizeof(int) * ndims);		//Input dimensions matrix, 2-D
		dims = 		(int*)malloc(sizeof(int) * ndims);	//Process topology dimensions
		period =	(int*)malloc(sizeof(int) * ndims);	//Periodicity for ndims dimensions (used in cartesian Topology)
		
///////////////////////////////////INPUT ROWS/COLS IN MATRIX//////////////////////////////

	if( rank == 0)			//If we remove rank == 0, we can simply set N[0], N[1] and avoid MPI_Bcast()
	{
	
		N[0] = PROB_SIZE;			//Number of rows that the domain is divided into
		N[1] = PROB_SIZE;			//Number of columns that the domain is divided into		
	}
	
///////////////////////////////////BROADCAST DIMENSIONS OF MATRIX/////////////////////////
	
	MPI_Bcast(N, ndims, MPI_INTEGER, 0, old_comm);		//If not aborted then broadcast value of N

///////////////////////////////////DECIDE TOPOLOGY OF PROCESSES ON RANK 0/////////////////////
	
	if(rank == 0)
	{
		if( (int)sqrt(P) * (int) sqrt(P) == P) //Check if no. of processes is a perfect square
		{
			fflush(stdout);
			dims[0] = (int) sqrt(P);
			dims[1] = (int) sqrt(P); 
		}
		else
		{
			i = (int)sqrt(P) + 1;					//Say P=8, then i = 3 
			 
			while( P % i != 0)						//Determining first i to divide P
				i++; 								//P=8 then i = 4
				
			for(j = 1 ; j <= (int)sqrt(P) ; j++)	//Determining which multiple of i divides P
				if( j * i == P)
					break;
			
			if( j > (int)sqrt(P) )
			{
				printf("\nNo topology can be decided for the %d number of processes", size); 
				MPI_Abort(old_comm, -1);
			}
				 
			dims[0] = i;							//No. of processes in Vertical direction
			dims[1] = j; 							//No. of processes in Horizontal direction
		}
		
		if((N[0] - 1) % dims[0] != 0)						//Check if rows divisible by number of processes 
		{
			printf("\nNumber of rows of unknown are not divisible by number of processes in dimension 1...Aborting");
			MPI_Abort(old_comm, -1);
		}
		
		if((N[1] - 1) % dims[1] != 0)						//Check if columns divisible by number of processes
		{
			printf("\nNumber of cols of unknown are not divisible by number of processes in dimension 2...Aborting");
			MPI_Abort(old_comm, -1);
		}		
		
		
	}
	
///////////////////////////////////BROADCAST DIMENSIONS OF PROCESS TOPOLOGY///////////////
	
	MPI_Bcast(dims, ndims, MPI_INTEGER, 0, old_comm);
	
	if( rank == 0)
		printf( "\nThe 2-D process topology is %d X %d ", dims[0], dims[1]); 
	
///////////////////////////////////ALLOCATE MATRICES ON EACH PROCESS//////////////////////

	prows = (N[0]-1)/dims[0];
	pcols = (N[1]-1)/dims[1]; 
	old = allocate_mem(prows+2, pcols+2);	//2 rows and 2 cols extra for receiving neighbours data 
	new = allocate_mem(prows+2, pcols+2); 	//Boundaries can be one of ghost rows/cols for some processes
	
////////////////////////////////////CREATE TOPOLOGY///////////////////////////////////////

	reorder = 0; 
	period[0] = 0;
	period[1] = 0;
	 
	MPI_Cart_create(old_comm, ndims, dims, period, reorder, &new_comm);
	
////////////////////////////////////FIND 4-NEAREST NEIGHBOURS/////////////////////////////

	direction = 0;									//First dimension i.e. row-up and row-down
	displacement = 1;								//Distance of 1 from current process
	
	MPI_Cart_shift(new_comm, direction, displacement, &up, &down);
	
	direction = 1; 									//Second dimension i.e. Left and right column
	displacement = 1;
	
	MPI_Cart_shift(new_comm, direction, displacement, &left, &right);
	
	printf("\nProcess %d neighbours: up = %d, down = %d, left = %d, right = %d", rank, up, down, left, right);
	printf("\n******************************************************************");
	
///////////////////////////////////SET BOUNDARIES TO 1 (or anything)/////////////////////////////////////

	if( up == MPI_PROC_NULL)									//If no up neighbour, set upper boundary to 1
	{
		for(j = 1; j <= pcols ; j++)
		{
			old[0][j] = 1;
			new[0][j] = 1;
		}
	}  
	
	if( down == MPI_PROC_NULL)									//If no down neighbour, set lower boundary to 1
	{
		for(j = 1; j <= pcols ; j++)
		{
			old[prows+1][j] = 1;
			new[prows+1][j] = 1;
		}
	}
	
	if( left == MPI_PROC_NULL)									//If no left neighbour, set leftmost column to 1
	{
		for(i = 1; i <= prows ; i++)
		{
			old[i][0] = 1;
			new[i][0] = 1;
		}
	}
	
	if(right == MPI_PROC_NULL)									//If no right neighbour, set rightmost boundary to 1
	{
		for(i = 1; i <= prows ; i++)
		{
			old[i][pcols+1] = 1;
			new[i][pcols+1] = 1;
		}
	}

//////////////////////////////////////DEFINE CONTIGUOUS DTYPE FOR UPPER/LOWER HALOS///////

	MPI_Type_contiguous(pcols, MPI_FLOAT, &row_type);
	MPI_Type_commit(&row_type);
	
//////////////////////////////////////DEFINE VECTOR DTYE FOR LEFT/RIGHT HALOS/////////////

	count = prows;									//Number of blocks
	blocklength = 1;								//Length of each block
	stride = pcols+2; 								//Distance from first element (including) to second element (excluding)
	
	MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &col_type);
	MPI_Type_commit(&col_type);
	
/////////////////////////////////////////WHILE LOOP SHOULD START HERE/////////////////////

while( G_max_err > Tol)
{
	
	iterations++;  

////////////////////////////////////////POST IRECV()//////////////////////////////////////
		
	MPI_Irecv(&old[0][1], 1, row_type, up, 0, new_comm, &recv[0]);			//From upper neighbour
	MPI_Irecv(&old[prows+1][1], 1, row_type, down, 0, new_comm, &recv[1]);	//From lower neighbour
	
	MPI_Irecv(&old[1][0], 1, col_type, left, 0,new_comm, &recv[2]);			//From left neighbour
	MPI_Irecv(&old[1][pcols+1], 1, col_type, right, 0, new_comm, &recv[3]);	//From right neighbour


/////////////////////////////////////////POST ISEND()/////////////////////////////////////

	MPI_Isend(&old[prows][1], 1, row_type, down,0, new_comm, &send[0]); 	//To lower neighbour
	MPI_Isend(&old[1][1], 1, row_type, up, 0, new_comm, &send[1]); 			//To upper neighbour
	
	MPI_Isend(&old[1][pcols], 1, col_type, right,0, new_comm, &send[2]);	//To right neighbour
	MPI_Isend(&old[1][1], 1, col_type, left, 0, new_comm, &send[3]) ;		//To left neighbour
	

////////////////////////////////////////INDEPENDENT UPDATE////////////////////////////////
	
	local_max_err = independent_update(old, new, prows+2, pcols+2);

/////////////////////////////////////////WAIT FOR DATA TO ARRIVE////////////////////////// 
	
	MPI_Waitall(4, recv, MPI_STATUSES_IGNORE);
	MPI_Waitall(4, send, MPI_STATUSES_IGNORE); 
	
///////////////////////////////////////DEPENDENT ROW UPDATE///////////////////////////////
	
	local_max_err_new = dependent_row_update(old, new, 1, pcols+2);
	
	if(local_max_err_new > local_max_err )
		local_max_err = local_max_err_new; 
	
	local_max_err_new = dependent_row_update(old, new, prows, pcols+2); 
	
	if(local_max_err_new > local_max_err )
		local_max_err = local_max_err_new; 

///////////////////////////////////////DEPENDENT COLUMN UPDATE////////////////////////////

	local_max_err_new = dependent_col_update(old, new, 1, prows+2);
	
	if(local_max_err_new > local_max_err )
		local_max_err = local_max_err_new; 
	
	local_max_err_new = dependent_col_update(old, new, pcols, prows+2); 
	
	if(local_max_err_new > local_max_err )
		local_max_err = local_max_err_new;
		
////////////////////////////////////////ALLREDUCE() FOR FINDING GLOBAL ERROR//////////////

	MPI_Allreduce(&local_max_err, &G_max_err, 1, MPI_FLOAT, MPI_MAX, new_comm );  
	
	if(rank == 0)
		printf("\n Global Error = %f",G_max_err);
	
///////////////////////////////////////FREE DATA TYPES/FINALIZE MPI///////////////////////
	
	temp = new ;
	new = old;
	old = temp;

	if(iterations % SAVE_ITER == 0)
		save_current_solution(new, prows, pcols, new_comm, N[0], iterations);  

}
	end = MPI_Wtime();
	if(rank == 0)
		printf("\nTotal time taken by program is = %lf\n", end-start);	 
	MPI_Type_free(&row_type);
	MPI_Type_free(&col_type);
	MPI_Finalize();
	return 0; 
}
