// Clanu 2017-2018 - INSA Lyon - GE

#include <iostream>
#include <cmath>
#include <cstring>
#include <windows.h>   // WinApi header
#include <iomanip>
#define FLOAT_TYPE float

#include "common_functions.h"
#include "clanu_functions.h"
#include "timing_functions.h"

#define COLOR_RESET 0x0F
#define FOREGROUND_WHITE 0x07
#define BACKGROUND_WHITE 0x70

using namespace std;

int main(int argc, char *argv[])
{
    if( argc < 4)
    {
        cerr << " Usage : " << argv[0] << " data\\path Test_filename.csv Theta.csv" << endl;
        return -1;
    }

    char test_file[50]  = {0};
    char theta_file[50] = {0};

    strcat(test_file , argv[1]);strcat(test_file , argv[2]);
    strcat(theta_file, argv[1]);strcat(theta_file, argv[3]);

    // Summarizing options
    cout << " ** summarize options : " << endl;
    cout << " \t Testing  file : " << test_file << endl;
    cout << " \t Theta  file : " << theta_file << endl;


    cout << "Reading and initializing ... This may take a while (~20-30s) " << endl;
    tic();

    // read TESTING CSV file
    FLOAT_TYPE **CSV=nullptr;
    unsigned int CSV_m, CSV_n;
    loadCSV_to_matrix( test_file, &CSV,  &CSV_m, &CSV_n);

    // Extract features X and labels y
    unsigned int m = CSV_m;
    unsigned int n = CSV_n - 1; // the first column contains the labels
    FLOAT_TYPE **X = nullptr; allocate( &X, m, n);
    FLOAT_TYPE  *y = nullptr; allocate( &y, m);

    extract_features_from_CSV( X, CSV, CSV_m, CSV_n );
    extract_labels_from_CSV  ( y, CSV, CSV_m );
    destroy( &CSV, CSV_m);
    normalize(X, CSV_m, CSV_n);

    // Allocate Theta variable
    FLOAT_TYPE **Theta=nullptr; allocate(&Theta, 10, n);
    unsigned int linesT, colsT;
    readTheta(theta_file, Theta, &linesT, &colsT);

    tac();
    cout << "Reading and initialization time : " << duration() << "s " << endl;


    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    int color;

    FLOAT_TYPE avg = 0;;
    FLOAT_TYPE c_avg[10] = {0};
    FLOAT_TYPE c_nb[10] = {0};
    FLOAT_TYPE *prob=nullptr; allocate(&prob, 10);
    FLOAT_TYPE max_prob;

    cout << "line\t|y\t|0\t|1\t|2\t|3\t|4\t|5\t|6\t|7\t|8\t|9\t|" << endl;
    for(unsigned int line = 0; line<m; line++)
    {
        unsigned int c_prob = 0;
        prob[0] = g( dot_product( Theta[0], X[line], n ) );
        max_prob = prob[0];
        for(unsigned int c=0; c<10; c++)
        {
            prob[c] = g( dot_product( Theta[c], X[line], n ) );
            if( max_prob < prob[c])
            {
                max_prob = prob[c];
                c_prob   = c;
            }
        }

        c_nb[(unsigned int)(y[line])]++;
        if( y[line] == c_prob ){
            avg++;
            c_avg[c_prob]++;
            color = FOREGROUND_GREEN|FOREGROUND_INTENSITY;
        }else{
            color = FOREGROUND_RED|FOREGROUND_INTENSITY;
        }

        SetConsoleTextAttribute(hConsole, COLOR_RESET);
        cout << line << "\t|" << y[line] << "\t|";
        for(unsigned int c=0; c<10; c++)
        {
            if(y[line] == c)
                SetConsoleTextAttribute(hConsole, BACKGROUND_BLUE|color);
            else
                SetConsoleTextAttribute(hConsole, color);
            cout << prob[c] << "\t";
            SetConsoleTextAttribute(hConsole, COLOR_RESET);
            cout << "|";
        }
        cout << endl;
    }

    SetConsoleTextAttribute(hConsole, COLOR_RESET);
    cout << "Digit accuracy:\t|";
    for(unsigned int c=0; c<10; c++)
    {
        c_avg[c] /= c_nb[c]/100;
        cout << fixed << setprecision(2) << c_avg[c] << "\t|";
    }
    cout << endl;
    avg /= m/100;
    cout << "Total accuracy:\t" << fixed << setprecision(2) << avg << endl;


    // free memory
    destroy( &prob   );
    destroy( &y      );

    destroy( &Theta, 10);
    destroy( &X, m     );

    cout << " end." << endl;
    return 0;
}
