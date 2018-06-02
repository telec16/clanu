// Clanu 2017-2018 - INSA Lyon - GE

#include <iostream>
#if defined(_OPENMP)
    #include <omp.h>
#endif

#define FLOAT_TYPE float

#include "common_functions.h"
#include "clanu_functions.h"
#include "timing_functions.h"
#include <cstring>

#define MAIN_LOOP 1
#define SUB_LOOP 2

using namespace std;

int main(int argc, char *argv[])
{
    // Test some compialtion options
    #if defined(_OPENMP)
        cout << " OPENMP is activated  : great! " << endl;
    #else
        cout << " OPENMP is not activated  (good for debug)" << endl;
    #endif

    #ifdef __FAST_MATH__
        cout << " fast-math is activated : great! " << endl;
    #else
        cout << " fast-math is strangely not activated " << endl;
    #endif

    FLOAT_TYPE tau_start;
    FLOAT_TYPE tau_step;
    FLOAT_TYPE tau_end;
    unsigned int max_it;

    cout << "argc:" << argc <<endl;
    if( argc < 9)
    {
        cerr << " Usage : " << argv[0] << " data\\path Train_filename.csv Test_filename.csv Theta_filename max_it tau_start tau_step tau_end" << endl;
        return -1;
    }

    char train_file[50] = {0};
    char test_file[50]  = {0};
    char theta_file[50] = {0};

    strcat(train_file, argv[1]);strcat(train_file, argv[2]);
    strcat(test_file , argv[1]);strcat(test_file , argv[3]);
    strcat(theta_file, argv[1]);strcat(theta_file, argv[4]);

    max_it      = stoul( argv[5] );
    tau_start   = stof ( argv[6] );
    tau_step    = stof ( argv[7] );
    tau_end     = stof ( argv[8] );
    // Summarizing options
    cout << " ** summarize options : " << endl;
    cout << " \t Training file : " << train_file << endl;
    cout << " \t Testing  file : " << test_file << endl;
    cout << " \t Theta  file : " << theta_file << endl;
    cout << " \t max_it = " << max_it << endl;
    cout << " \t tau    = " << tau_start << ":" << tau_step << ":" << tau_end << endl;

    cout << "Reading and initializing ... This may take a while (~20-30s) " << endl;
    tic();

    // read TRAINING CSV file
    FLOAT_TYPE **CSV=nullptr;
    unsigned int CSV_m, CSV_n;
    loadCSV_to_matrix( train_file, &CSV,  &CSV_m, &CSV_n);

    // Extract features X and labels y
    unsigned int m = CSV_m;
    unsigned int n = CSV_n - 1; // the first column contains the labels
    FLOAT_TYPE **X = nullptr; allocate( &X, m, n);
    FLOAT_TYPE  *y = nullptr; allocate( &y, m);

    extract_features_from_CSV( X, CSV, CSV_m, CSV_n );
    extract_labels_from_CSV  ( y, CSV, CSV_m );
    destroy( &CSV, CSV_m);

    // Read TESTING CSV file
    loadCSV_to_matrix( test_file, &CSV,  &CSV_m, &CSV_n);

    // Extract features test_X and labels test_y
    unsigned int test_m = CSV_m;
    FLOAT_TYPE **test_X = nullptr; allocate( &test_X, m, n);
    FLOAT_TYPE  *test_y = nullptr; allocate( &test_y, m);

    extract_features_from_CSV( test_X, CSV, test_m, CSV_n );
    extract_labels_from_CSV  ( test_y, CSV, test_m );
    destroy( &CSV, CSV_m);


    // Allocate Theta variable
    FLOAT_TYPE **Theta=nullptr; allocate(&Theta, 10, n);
    FLOAT_TYPE **Theta_max=nullptr; allocate(&Theta_max, 10, n);

    // Training : Defined variables
    FLOAT_TYPE cumulative_error;
    FLOAT_TYPE max_acc;
    FLOAT_TYPE **grad_J=nullptr; allocate(&grad_J,10,n);                    // (Used as memory) Create and allocate the gradient of J (vector of n elements) for each digit (10)
    FLOAT_TYPE **d_k=nullptr; allocate(&d_k,10,n);                          // (Used as memory) Create and allocate the direction d for each digit

    tac();
    cout << "Reading and initialization time : " << duration() << "s " << endl;

    tic(MAIN_LOOP);
    for(FLOAT_TYPE tau=tau_start; tau<=(tau_end+.01); tau+=tau_step)
    {
        zeros(Theta, 10, n);
        zeros(Theta_max, 10, n);
        ones(grad_J, 10, n);                                                // To prevent divide by zero, as much as we like indian food, this is not wanted !
        zeros(d_k,10,n);
        max_acc = 0;

        cout<<"Tau="<<tau<<endl;

        // Training : k iterations
        tic(SUB_LOOP);
        for(unsigned int k=0; k < max_it; k++)
        {
            cumulative_error = 0;
            tic();
            // Training : c digits
    #if defined(_OPENMP)
        #pragma omp parallel for reduction(+:cumulative_error)  // Concurrency (or parallel) for loop
    #endif
            for(unsigned int c=0; c<10; c++)
            {
                FLOAT_TYPE *theta_c_k = Theta[c];                    		//  linked on Theta, for easier reading
                FLOAT_TYPE *d_c_k=nullptr; allocate(&d_c_k,n);       		// ( definied and allocated here for concurrency )
                FLOAT_TYPE *grad_J_c_k=nullptr; allocate(&grad_J_c_k,n); 	// create and allocate gradient of J for one column
                FLOAT_TYPE *temp=nullptr; allocate(&temp,n); 				// define a temporary variable
                FLOAT_TYPE beta_c_k;                                        // d_c_k = {0,0,0...} i.e. initialization

                /*- GRAD(J) -*/
                //grad_J_c_k = sum(X .* (sigmoid(X * theta_c_k)-y_c_i))
    #if defined(_OPENMP)
        #pragma omp parallel for reduction(+:cumulative_error)  // Concurrency (or parallel) for loop
    #endif
                for(unsigned int i=0; i<m; i++)
                {
                    FLOAT_TYPE y_c_i = (y[i]==c)?1.0:0.0;            		// y_c_i = 1 if y[i] == c, 0  otherwise
                    FLOAT_TYPE h_theta_c_i =
                            g( dot_product( theta_c_k, X[i], n ) ) - y_c_i; // h_theta_c_i = g( theta_c_k . X[i] ) - y_c_i
                    mac_v_v_s( grad_J_c_k, X[i], h_theta_c_i, n );    		//  grad_J_c_k += X[i] * h_theta_c_i

                    cumulative_error += abs(h_theta_c_i);            		// ( used for evolution tracking )
                }
                mul_v_s( grad_J_c_k, grad_J_c_k, 1.0/m, n);					// grad_J_c_k = grad_J_c_k * 1/m -> gradient of J(theta c,k)

                /*- BETA -*/
                //beta_c_k = (grad(J_c_k) . (grad(J_c_k)-grad(J_c_k-1))) / (norm(grad(J_c_k-1))^2)
                sub_2v(temp,grad_J_c_k,grad_J[c],n); 						// temp = grad(J_c_k) - grad(J_c_k-1)
                beta_c_k  = dot_product(grad_J_c_k,temp,n);                 // beta_c_k = grad(J_c_k) . temp
                beta_c_k /= norm_v_sqr(grad_J[c],n);                        // beta_c_k = beta_c_k/sum(grad(J_c_k-1)^2)
                memcpy(grad_J[c], grad_J_c_k, sizeof(FLOAT_TYPE) * n);		// grad(J_c_k-1) = grad(J_c_k)

                /*- DIRECTION -*/
                //d_c_k = (beta_c_k * d_c_k-1) - grad(J_c_k)
                if(k != 0)
                    mul_v_s(d_c_k, d_k[c], beta_c_k,n);                     // d_c_k = beta_c_k * d_c_k-1
                else
                    zeros(d_c_k,n);
                sub_2v(d_c_k,d_c_k,grad_J_c_k,n);                           // d_c_k = d_c_k - grad(J_c_k)
                memcpy(d_k[c], d_c_k, sizeof(FLOAT_TYPE) * n);              // d_c_k-1 = d_c_k

                /*- UPDATING THETA -*/
                //theta_c_k = theta_c_k + tau * d_c_k
                mul_v_s(d_c_k,d_c_k,tau,n);									// d_c_k = tau * d_c_k
                sum_2v( theta_c_k, theta_c_k, d_c_k, n);        			// theta_c_k = theta_c_k + d_c_k


                destroy(&d_c_k);
                destroy(&grad_J_c_k);
                destroy(&temp);
            }
            tac();
            cout << "it : " << k << "\t time : " << duration() << " s\t error : "<< cumulative_error/(10*m);

            FLOAT_TYPE acc = Accuracy(Theta, test_X, test_y, m, n)*100;
            if((k%5) == 0)
            {
                cout << "\taccuracy : "<<acc;
            }
            if(max_acc < acc)
            {
                saveTheta("temp", Theta, 10, n);
                //copy_M(Theta_max, Theta, m, n);
                cout << "\tmax : "<<acc;
                max_acc = acc;
            }
            cout << endl;
        }
        unsigned int linesT, colsT;
        readTheta("temp", Theta_max, &linesT, &colsT);

        FLOAT_TYPE facc = Accuracy(Theta_max, test_X, test_y, m, n)*100;
        cout << "Final accurracy : " << facc << endl;

        char file[80] = {0};
        sprintf(file, "%s_tau=%.1f_maxIt=%d_acc=%.3f.csv", theta_file, tau, max_it, facc);
        saveTheta(file, Theta_max, 10, n);

        tac(SUB_LOOP);
        cout << "training time : " << duration(SUB_LOOP) << endl;
    }

    tac(MAIN_LOOP);
    cout << "total time : " << duration(MAIN_LOOP) << endl;

    // free memory
    cout << "Freeing :" << endl;
    cout << "-y" << endl;
    destroy( &y      );
    cout << "-test_y" << endl;
    destroy( &test_y );

    cout << "-Theta" << endl;
    destroy( &Theta, 10);
    cout << "-X" << endl;
    destroy( &X, m     );
    cout << "-test_X" << endl;
    destroy( &test_X, test_m );

    cout << "-grad_J" << endl;
    destroy( &grad_J, 10);
    cout << "-d_k" << endl;
    destroy( &d_k, 10);

    cout << " end." << endl;
    return 0;
}
