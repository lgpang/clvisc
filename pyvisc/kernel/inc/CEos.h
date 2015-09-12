#ifndef __CEOS__
#define __CEOS__

#include<iostream>
#include<fstream>
#include<algorithm>
#include<vector>
#include<string>
#include<cmath>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>


using namespace std;

typedef const double CD ;


/*! \breif The simple spline class for EOS interpolation */
class CSpline{
    private:
        gsl_interp_accel * acc;
        gsl_spline * spline;
        int SIZE;
        bool is_alloced;
    public:
        /*! \breif HIGH/fHIGH=the maximum x/y; LOW/fLOW=the minimum x/y */
        double HIGH, LOW;
        double fLOW, fHIGH;
        double HIGHx0, HIGHy0; //for extrapolation when exceed maximum

        CSpline(){
            is_alloced = 0;
        }

        void alloc(double * x, double * y, int length ){
            acc = gsl_interp_accel_alloc();
            spline = gsl_spline_alloc( gsl_interp_cspline, length );
            gsl_spline_init( spline, x, y, length );

            is_alloced = 1;

            SIZE = length ;
            HIGH = x[ length-1 ];
            fHIGH =y[ length-1 ];
            LOW  = x[ 0 ];
            fLOW = y[ 0 ];

            HIGHx0 = x[length-2];
            HIGHy0 = y[length-2];
        }

        /*! \brief Initialize cspline with two arrays x, y and the length */
        CSpline( double * x, double * y, int length ){
            alloc( x, y, length );
        }

        double lin_int(CD &x1, CD &x2, CD &y1, CD &y2, CD &x){
            double r= (x-x1)/(x2-x1);
            return y1*(1-r) + y2*r;
        }


        /*! \breif Evaluate the function value at x1
         *  \return The value from cspline interpolation at point x1 */
        double eval( const double & x1 ){
            if( x1>= LOW && x1 <= HIGH )
                return gsl_spline_eval( spline, x1, acc );
            else if( x1 < LOW ){
                //cerr<<"Error:"<<x1<<" not in range:["<<LOW<<','<<HIGH<<"]\n";
                return lin_int( 0.0, LOW, 0.0, fLOW, x1 );
            }
            else if( x1 > HIGH ){
                return lin_int( HIGHx0, HIGH, HIGHy0, fHIGH, x1 );
                //cerr<<"Error:"<<x1<<" not in range:["<<LOW<<','<<HIGH<<"]\n";
            }
        }

        ~CSpline(){
            if( is_alloced ){
                gsl_spline_free(spline);
                gsl_interp_accel_free(acc);
            }
        }
};


class CEos{
    public:
        virtual void InitEos()=0;
        /** \return Pressure in units of GeV/fm^3 */
        virtual double P( const double & ed, const double & mu ) = 0;
        /** \return Temperature in units of GeV */
        virtual double T( const double &  ed, const double & mu ) = 0;
        /** \return Entropy density in units of fm^{-3} */
        virtual double S( const double &  ed, const double & mu ) = 0;
        /** \return Energy density in units of GeV/fm^3 */
        virtual double Ed( const double &  T, const double & mu ) = 0;
        /** \return QGP fraction in units of 1 */
        virtual double F_T( const double &  T, const double & mu ) = 0;
        /** \return QGP fraction in units of 1 */
        virtual double F_ed( const double &  ed, const double & mu ) = 0;
};

class CEosI:public CEos{
    private:
        double dof;   // effective quark gluon degree of freedom
        double hbarc, hbarc3;
        double coef; 
    public:
        CEosI(){ 
            InitEos(); 
        }

        void InitEos(){
            dof = 169.0 / 4.0;   
            hbarc = 0.1973269631;
            hbarc3 = pow(0.1973269631, 3);
            coef = M_PI*M_PI/30.0;
        }

        double P( const double & ed, const double & mu ){
            return ed/3.0;
        }

        double T( const double &  ed, const double & mu ){
            return hbarc*pow( 1.0/(dof*coef)*ed/hbarc, 0.25 );
        }

        double S( const double &  ed, const double & mu ){
            return ( ed + P(ed, mu) )/T( ed, mu );
        }

        double Ed( const double &  T, const double & mu ){
            return dof*coef * T*T*T*T/hbarc3 ;
        }
        double F_T( const double &  T, const double & mu ) {
            return 1.0;
        }
        double F_ed( const double &  ed, const double & mu ){
            return 1.0;
        }
};

class CEosL:public CEos{
    private:
        vector<double> VEd;
        vector<double> VP;
        vector<double> VT;
        vector<double> VS;
        vector<double> VF;
        int SIZE;
        string EosType;

        /*!\breif 1.Ped: P as a function of ed
         *        2.Ted: T as a function of ed
         *        3.Sed: S  as a function of ed
         *        5.Fed: QGP fraction as a function of ed
         *        4.EdT: Ed as a function of T
         *        6.FT:  QGP fraction as a function of T
         */
        CSpline Ped, Ted, Sed, EdT, Fed, FT;

    public:
        CEosL(const string & EosType){
            this->EosType = EosType;
            InitEos();
        }

        void InitEos(){
            char fname[256];
            char fname1[256];
            /*!\breif read in data file */
            for( int i=4; i>=1; i-- ){
                if( EosType == "PCE" ){
                    sprintf( fname, "../EOSTable/s95p-PCE165-v0/s95p-PCE165-v0_dens%d.dat", i );
                    sprintf( fname1, "../EOSTable/s95p-PCE165-v0/s95p-PCE165-v0_par%d.dat", i );
                }
                else{ // EosType == "CE" or anything else
                    sprintf( fname, "../EOSTable/s95p-v1/s95p-v1_dens%d.dat", i );
                    sprintf( fname1, "../EOSTable/s95p-v1/s95p-v1_par%d.dat", i );
                }

                ifstream fin( fname  );
                ifstream fin1( fname1 );
                if( !fin.is_open() || !fin1.is_open() ){
                    cerr<<"Open EOS data file failed! \n";
                }
                else{
                    double e0, de, ne;
                    fin>>e0>>de>>ne;
                    fin1>>e0>>de>>ne;

                    double Ed, Pr, S, Temp, n, frac, mu, tmp;
                    for(int j=0; j<ne; j++){
                        fin>>Ed>>Pr>>S>>n>>frac;
                        fin1>>Temp>>mu>>tmp;
                        //Temp = (Ed + Pr)/S ;
                        if( j!= 0 ){ //double counting for j=0
                            VEd.push_back( Ed );
                            VP.push_back( Pr );
                            VT.push_back( Temp );
                            VS.push_back( S );
                            VF.push_back( frac );
                        }
                    }
                }
                fin.close();
                fin1.close();
            }

            reverse( VEd.begin(), VEd.end() );
            reverse( VP.begin(), VP.end() );
            reverse( VT.begin(), VT.end() );
            reverse( VS.begin(), VS.end() );
            reverse( VF.begin(), VF.end() );


            /*! \breif construct EOS by gsl_spline interpolation */
            
            SIZE = VEd.size();


            Ped.alloc( VEd.data(), VP.data(), SIZE );
            Ted.alloc( VEd.data(), VT.data(), SIZE );
            Sed.alloc( VEd.data(), VS.data(), SIZE );
            Fed.alloc( VEd.data(), VF.data(), SIZE );
            EdT.alloc( VT.data(), VEd.data(), SIZE );
            FT.alloc(  VT.data(), VF.data(),  SIZE );
        }

        double T( const double & ed, const double & mu ){
            if( ed <= VEd.at(SIZE-1) ){
                return Ted.eval( ed );
            }
            else{
                return (ed + P( ed, 0.0 ))/S( ed, 0.0 );
            }
        }

        double P( const double & ed, const double & mu ){

            if( ed <= VEd.at(SIZE-1) ){
                return Ped.eval( ed );
            }
            else{
                return 0.3327*ed-0.3223*pow(ed,0.4585)-0.003906*ed*exp(-0.05697*ed);
            }
        }

        double S( const double & ed, const double & mu ){
            if( ed <= VEd.at(SIZE-1) ){
                return Sed.eval( ed );
            }
            else{
                double s=18.202*ed-62.021814-4.85479*exp(-2.72407*1.0E-11*pow(ed,4.54886))\
            +65.1272*pow(ed,-0.128012)*exp(-0.00369624*pow(ed,1.18735));
                return pow(s, 3.0/4.0);
            }
        }

        double Ed( const double & T, const double & mu ){
            return EdT.eval( T );
        }

        double F_ed( const double & ed, const double & mu){
            return Fed.eval( ed );
        }

        double F_T( const double & T, const double & mu ){
            return FT.eval( T );
        }

};

//#define test_main
#ifdef test_main

int main(int argc, char** argv)
{

CEos *Eos, *Eos1;
Eos = new CEosL("PCE");
Eos1 = new CEosI();

cout<<"For PCE Eos"<<endl;
cout<<"P(30.0) = "<<   Eos->P( 30.0, 0.0 )<<endl;
cout<<"T(0.120) = "<<  Eos->T( 0.120, 0.0 )<<endl;
cout<<"Ed(0.136) = "<< Eos->Ed( 0.136, 0.0 )<<endl;
cout<<"Ed(0.120) = "<< Eos->Ed( 0.120, 0.0 )<<endl;

cout<<"For Ideal Eos"<<endl;
cout<<"P(0.120) = "<< Eos1->P( 0.120, 0.0 )<<endl;
cout<<"T(0.120) = "<< Eos1->T( 0.120, 0.0 )<<endl;
return 0;

}

#endif

#endif
