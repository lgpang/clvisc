#ifndef CCUBE_H
#define CCUBE_H
#include<vector>
#include<algorithm>
#include<iostream>
#include"ran2.h"
#include<iomanip>
#include<cmath>
#include<ctime>
#include<cstdlib>
#include"PhyConst.h"
using namespace std;
typedef const int CI;
typedef const double CD;

typedef CMatrix_3D <double> M3D;

/*4D point class */
class Point{
    friend Point operator-(const Point & p1, const Point & p2);
    public:
    double t;
    double x;
    double y;
    double z;
    Point():t(0.0),x(0.0),y(0.0),z(0.0){}    //default construction
    Point(CD &t1, CD &x1, CD &y1, CD &z1):t(t1),x(x1),y(y1),z(z1){}
    Point(const Point &p1){
	t=p1.t;
	x=p1.x;
	y=p1.y;
	z=p1.z;
    }//Copy construction

    bool operator==(const Point &rhs){
	return (t==rhs.t && x==rhs.x && y==rhs.y && z==rhs.z);
    }//judge if 2 points are the same

    void printout(){
	cout<<"("<<setw(8)<<t
	    <<","<<setw(8)<<x
	    <<","<<setw(8)<<y
	    <<","<<setw(8)<<z
	    <<")"<<endl;
    }//print out this point
};
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
/* This is used to set the 16 corners' energy density and velocity  */
class Corner{
    public:
	int id;   //Corner id;
	Point pos;//Corner position;
	double value;//Corner energy density 
	double VL[3];//Corner velocity
	double pi[10];

	Corner(){}   //Default construction
	Corner(CI & idx, Point& cord, CD & val):id(idx), pos(cord), value(val){}
	Corner(const Corner & c1){
	    id=c1.id;
	    pos=c1.pos;
	    value=c1.value;
	}
};
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
/* This is used to set the 32 edges's information of the 4D cube, which 
will be cut by the freeze out hyper surface                         */

class Edge{
    public:
	int id;
	Corner lc;
	Corner rc;
	Edge(){}
	Edge(CI & idx, Corner & c1, Corner &c2):id(idx), lc(c1), rc(c2){};
	Edge(const Edge & e1){
	    id=e1.id;
	    lc=e1.lc;
	    rc=e1.rc;
	}
	Point MiddlePoint(){
	    Point mp( 0.5*(lc.pos.t+rc.pos.t), 0.5*(lc.pos.x+rc.pos.x), 0.5*(lc.pos.y+rc.pos.y), 0.5*(lc.pos.z+rc.pos.z) );
	    return mp;
	}
};

//////////////////////////////////////////////////////////////////////
/** Store the interpolation position on the edges      **************/
class CIntersection{
    public:
	int IEdge;
	Point pos;
	CIntersection(){}
	CIntersection(int IE, Point pt):IEdge(IE),pos(pt){}
	CIntersection(const CIntersection & rhs){
	    IEdge=rhs.IEdge;
	    pos=rhs.pos;
	}
	bool operator==(const CIntersection & rhs){
	    return (IEdge==rhs.IEdge);
	}
};

typedef const CIntersection CCI;

//////////////////////////////////////////////////////////////////////
// A HyperSurface (one tetrahedra) of the 4-simplex in 4D space.    //
//////////////////////////////////////////////////////////////////////
class CHyperSurf{
    friend bool BeEqual(const CHyperSurf & lhs, const CHyperSurf & rhs);
    public:
    int ID;      //ID=The vertex ID which is oppsite to the facet.
    Point Vhs;   //normal vector of the hypersurface, toward outside
    Point Center;  //The center of the hypersurface
    int IntsID[4]; //The ints' ID for one hypersf
    bool BRemoved; //if the facet is used by two simplex, BRemoved=true
    CHyperSurf(){}
};
//////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////
/* A 4-simplex in 4D space. Notice a triangle is a 2-simplex in 2D space,
A tetrahedra is a 3-simplex in 3D space.                         */

class C4simplex{
    public:
	CIntersection ints[5];  // The 5 vertices of the 4-simplex, ID=array index
	CHyperSurf    hsuf[5];  // The 5 surface of the 4-simplex, ID=array index
	C4simplex(){}

	C4simplex(CIntersection Ints[5]){
	    for(int i=0; i<5; i++)ints[i]=Ints[i];
	    UpdateHyperSurf(); //update hypersurface after vertex changed
	}
	void CopyInts(const C4simplex & oldSimplex){
	    for(int i=0; i<5; i++)
		ints[i] = oldSimplex.ints[i];
	}// Copy one old 4-simplex to a new one
	void TinyMove(long & NSEED, const int& NTinyMove){
	    /* Move the 5 vertices a little bit to make them not cor-plannar, 
	     * this will not affect final results too much */
	     //cout<<"NSEED="<<NSEED<<endl;
	    for(int i=0; i<5; i++){
		ints[i].pos.t +=1.0E-9*(ran2(&NSEED)-0.5);
		ints[i].pos.x +=1.0E-9*(ran2(&NSEED)-0.5);
		ints[i].pos.y +=1.0E-9*(ran2(&NSEED)-0.5);
		ints[i].pos.z +=1.0E-9*(ran2(&NSEED)-0.5);
	    }
	    //cout<<"Moved once!"<<endl;
	}

	void UpdateHyperSurf(){   //update hypersurface after vertex changed
	    Point normv;          //normal vector of the hypersurface
	    Point vout;           //vector towards outside of the 4-simplex;
	    int NTinyMove=0;
UpdateAfterTinyMove:
	    for(int i=0; i<5; i++){
		hsuf[i].ID=i;
		Point center_4;       //center of one hypersurface;
		Point psurf[4];       //4 vertices for one hypersurface;

		int k=0;
		for(int j=0;j<5;j++){
		    if(j!=i){
			hsuf[i].IntsID[k] = ints[j].IEdge;
			center_4.t += 0.25*ints[j].pos.t;
			center_4.x += 0.25*ints[j].pos.x;
			center_4.y += 0.25*ints[j].pos.y;
			center_4.z += 0.25*ints[j].pos.z;
			psurf[k] = ints[j].pos; //select the hypersurface's 4 vertices
			k ++;
		    }
		}
		vout = center_4 - ints[i].pos;      

		Point v1=psurf[1]-psurf[0];
		Point v2=psurf[2]-psurf[0];
		Point v3=psurf[3]-psurf[0];
		normv.t = ( v1.x*(v2.y*v3.z-v2.z*v3.y) + v1.y*(v2.z*v3.x-v2.x*v3.z) + v1.z*(v2.x*v3.y-v2.y*v3.x) );
		normv.x = -( v1.t*(v2.y*v3.z-v2.z*v3.y) + v1.y*(v2.z*v3.t-v2.t*v3.z) + v1.z*(v2.t*v3.y-v2.y*v3.t) );
		normv.y = ( v1.t*(v2.x*v3.z-v2.z*v3.x) + v1.x*(v2.z*v3.t-v2.t*v3.z) + v1.z*(v2.t*v3.x-v2.x*v3.t) );
		normv.z = -( v1.t*(v2.x*v3.y-v2.y*v3.x) + v1.x*(v2.y*v3.t-v2.t*v3.y) + v1.y*(v2.t*v3.x-v2.x*v3.t) );

		//cout<<"normv="; normv.printout();
		//cout<<"vout ="; vout.printout();

		double Dvn=vout.t*normv.t + vout.x*normv.x + vout.y*normv.y + vout.z*normv.z ;
		//assert(Dvn!=0);//cor-plannar
		if(Dvn == 0){
		    TinyMove(NSEED, NTinyMove);
		    NTinyMove ++;
		    if(NTinyMove < 5){
			goto UpdateAfterTinyMove;
		    }//If the 5 points cor-planar in 4D space, move them a little bit randomly
		    else{
			cout<<"NTinyMove="<<NTinyMove<<endl;
			for(int l=0; l<5; l++)ints[l].pos.printout();
			assert(NTinyMove<4);
		    }//If tiny move happened too many times, stop the program
		}
		if(Dvn < 0){
		    normv.t = - normv.t;
		    normv.x = - normv.x;
		    normv.y = - normv.y;
		    normv.z = - normv.z;
		}//We settle the norm vector of the hyper surface toward out side of the 4-simplex
		hsuf[i].Vhs = normv;                //hyper surface's norm vector
		hsuf[i].Center = center_4;          //hyper surface's center point
		hsuf[i].BRemoved = false;           //If it's a inner hyper surface, BRemoved=true

		vector<int> vint(hsuf[i].IntsID, hsuf[i].IntsID+4);  
		sort(vint.begin(), vint.begin()+4);                
		int j=0; 
		for(vector<int>::iterator it=vint.begin(); it!=vint.end(); ++it){
		    hsuf[i].IntsID[j]=*it;
		    j++;
		}//sort the hsuf's 4 corners' id so that we can compare if two hyper-surface are the same one
	    }
	}//end update
};

class Cube{
    public:
	Point MassCenter;  //mass center of all the interpolation points
	Point V;           //velocity vt, vx, vy, vetas at MassCenter
	Point Vsurf;       //vector that towards low energy density direction
	double pi[10];     //pi^{tt tx ty te xx xy xe yy ye ee} 
	Point Origin;      //left bottom corner (tau, x, y, z) of the 4D cube
	Corner corners[16];//16 corners for the 4 dimensional cube
	Edge   edges[32];  //32 edges for the 4 dimensional cube
	double IntsValue;    //The Edec or Tdec that will make interpolation points on the edges.

	int Nints;       //Number of intersection points
	CIntersection *ints; //Array of intersection points

	vector<C4simplex> simps;    //4-simplexs constructed in the 4D cube


	double DTD, DXD, DYD, DZD;   //The size of the interpolation cube

	Cube();
	Cube(double IValue, double dtd, double dxd, double dyd, double dzd);
    /* IValue is Edec or Tdec, DTD is NTD*dt, DXD=NXD*dx, DYD=NYD*dy, DZD=NZD*detas */
    
	~Cube(){delete [] ints;}

	bool IsEdge(const Corner &c1, const Corner &c2);//check if 2 corners are on the same edge
	double Distance2(const Point &p1, const Point &p2);//calc the distance^2 of 2 points
	void SetOrigin(CD & t, CD & x, CD & y, CD & z);    //Set the position of the interpolation cube
	void SetCornersValue(double ED_corners[16]);       //Set the energy density of the 16 corners
	void SetCornersVelocity(double VLCorners[16][3]);  //Set the velocity vx, vy, vetas of the 16 corners
	void SetCornersPimn(double PICorners[16][10]); //Set corners pimn to calculate the pimn at mass center
	void CalcInts();             //Calc the Edges and Positions where the hyper surface cutting through the 4D cube
	void CalcVelocity();         //clac velocity of the hyper surfaces throuth 4D interpolation 

	void ConsNew_4simplex(CCI & newi, CI & ID);/*Constuct a new 4-simplex with new cut points and the old 4-simplex*/
	void Remv_Inner_SF(CI & lhs, CI &rhs);     /*Remove the inner hyper surface of the hyper volume */
	void GetAll_C4simplex();                   /*As this function name */
	void CalcSF2(double &DA0, double &DA1, double &DA2, double &DA3);
	//calc hypersurface d\Sigma_\mu and choose the hypersurface direction to make (V \cdot dS_u)>0
};

#endif
