#include<CCube.h>

long C4simplex::nseed = 356111111;

Point operator-(const Point & p1, const Point & p2)
{
	Point tmp;
	tmp.t=p1.t-p2.t;
	tmp.x=p1.x-p2.x;
	tmp.y=p1.y-p2.y;
	tmp.z=p1.z-p2.z;
	return tmp;
}//Define a vector by p1-p2

bool BeEqual(const CHyperSurf & lhs, const CHyperSurf & rhs)
{
/*Notice that the hyper surfaces' 4 corners are sorted in increasing order*/
	for(int i=0; i<4; ++i){
		if(lhs.IntsID[i] != rhs.IntsID[i])
			return false;
	}
	return true;
}//To judge if one hypersf was used twice by two new produced simplex

void Cube::Remv_Inner_SF(CI & lhs, CI &rhs)
{
	for(int i=0; i<5; i++)
		for(int j=0; j<5; j++){
			if( BeEqual(simps[lhs].hsuf[i], simps[rhs].hsuf[j]) ){
				simps[lhs].hsuf[i].BRemoved = true;
				simps[rhs].hsuf[j].BRemoved = true;
			}
		}
}//Remove Inner surface that used by two 4-simplexs

bool Cube::IsEdge(const Corner &c1, const Corner &c2){
	double distance=abs(c1.pos.t-c2.pos.t)+abs(c1.pos.x-c2.pos.x)+abs(c1.pos.y-c2.pos.y)+abs(c1.pos.z-c2.pos.z);
	if(distance==1.0) return true;
	else return false;
}// To see if the connection of two corners is one edge of the 4D cube

double Cube::Distance2(const Point &p1, const Point &p2){
	double dt2=(p2.t-p1.t)*(p2.t-p1.t);
	double dx2=(p2.x-p1.x)*(p2.x-p1.x);
	double dy2=(p2.y-p1.y)*(p2.y-p1.y);
	double dz2=(p2.z-p1.z)*(p2.z-p1.z);
	return (dt2+dx2+dy2+dz2);
}// The distance square of two points

void Cube::SetCornersValue(double ED_corners[16]){
	double VL[4]={0.0};
	double VH[4]={0.0};
	double ELSUM=0.0;
	double EHSUM=0.0;

	for(int n=0; n<2; n++)
		for(int i=0; i<2; i++)
			for(int j=0; j<2; j++)
				for(int k=0; k<2; k++)
				{
					int ID=8*n+4*i+2*j+k;
					corners[ID].value = ED_corners[ID];//Set corners' energy density

					double DEK = IntsValue-corners[ID].value;
					double ADEK= abs(DEK);
					if( DEK>0 ){
						ELSUM += ADEK;
						if(n==1) VL[0] += ADEK;
						if(i==1) VL[1] += ADEK;
						if(j==1) VL[2] += ADEK;
						if(k==1) VL[3] += ADEK;
					}
					else{
						EHSUM += ADEK;
						if(n==1) VH[0] += ADEK;
						if(i==1) VH[1] += ADEK;
						if(j==1) VH[2] += ADEK;
						if(k==1) VH[3] += ADEK;
					}
				}
	if(abs(ELSUM) > 1.0e-15){
		VL[0] = VL[0]/ELSUM;    
		VL[1] = VL[1]/ELSUM; 
		VL[2] = VL[2]/ELSUM; 
		VL[3] = VL[3]/ELSUM; 
	}
	if(abs(EHSUM) > 1.0e-15){
		VH[0] = VH[0]/EHSUM;
		VH[1] = VH[1]/EHSUM;
		VH[2] = VH[2]/EHSUM;
		VH[3] = VH[3]/EHSUM;
	}

/*Vsurf is the normalized vector towards low energy density*/
	Vsurf.t = VL[0] - VH[0];
	Vsurf.x = VL[1] - VH[1];
	Vsurf.y = VL[2] - VH[2];
	Vsurf.z = VL[3] - VH[3];

	for(int i=0; i<32; i++){
		int id1=edges[i].lc.id;
		int id2=edges[i].rc.id;
		edges[i].lc.value = corners[id1].value;
		edges[i].rc.value = corners[id2].value;
	}
}// Set corners value on the edges and calc the direction of hypersurface

void Cube::SetCornersVelocity(double VLCorners[16][3])
{
	for(int i=0; i!=16; i++)
		for(int j=0; j!=3; j++)
			corners[i].VL[j]=VLCorners[i][j];//set corner velocity
}
void Cube::SetCornersPimn(double PICorners[16][10])
{
 //Set corners pimn to calculate the pimn at mass center
  for(int i=0; i!=16; i++)
    for(int j=0; j!=10; j++)
      corners[i].pi[j] = PICorners[i][j];
}


void Cube::CalcInts(){
	Nints = 0;
	for(int l=0; l<32; l++){
		double dE1 = IntsValue-edges[l].lc.value;              //Edec - E0
		double dE2 = IntsValue-edges[l].rc.value;              //Edec - E1
		double dE12= edges[l].lc.value - edges[l].rc.value;    //E0-E1
		if( dE1*dE2 < 0 ){
			double intsp= abs(dE1)/abs(dE12);         //interpolation point on the edge
			ints[Nints].IEdge = l;
			ints[Nints].pos.t = ((edges[l].lc.pos.t - edges[l].rc.pos.t)==0.0) ?  edges[l].lc.pos.t : intsp ;
			ints[Nints].pos.x = ((edges[l].lc.pos.x - edges[l].rc.pos.x)==0.0) ?  edges[l].lc.pos.x : intsp ;
			ints[Nints].pos.y = ((edges[l].lc.pos.y - edges[l].rc.pos.y)==0.0) ?  edges[l].lc.pos.y : intsp ;
			ints[Nints].pos.z = ((edges[l].lc.pos.z - edges[l].rc.pos.z)==0.0) ?  edges[l].lc.pos.z : intsp ;
			Nints ++;
		}
	}
	MassCenter = Point(0.0,0.0,0.0,0.0);
	for(int i=0; i<Nints; i++){
		MassCenter.t += ints[i].pos.t / double(Nints);
		MassCenter.x += ints[i].pos.x / double(Nints);
		MassCenter.y += ints[i].pos.y / double(Nints);
		MassCenter.z += ints[i].pos.z / double(Nints);
	}//Calc the center of all the intersection points

}// Calc intersections


void Cube::ConsNew_4simplex(CCI & newi, CI & ID)
	/*Constuct new 4-simplex with new intersect and old 4-simplex,
	 * ID is the index for old simplex */
{
	Point vextra;
	for(int i=0; i<5; i++){
		C4simplex spx;
		if(simps[ID].hsuf[i].BRemoved==false){
			vextra = newi.pos - simps[ID].hsuf[i].Center;
			//note vextra is a contravariant vector, Vhs is a corvariant vector
			if( vextra.t*simps[ID].hsuf[i].Vhs.t
					+vextra.x*simps[ID].hsuf[i].Vhs.x
					+vextra.y*simps[ID].hsuf[i].Vhs.y
					+vextra.z*simps[ID].hsuf[i].Vhs.z > 0){
				spx.CopyInts(simps[ID]);
				spx.ints[i]=newi;
				spx.UpdateHyperSurf();
				spx.hsuf[i].BRemoved=true;
				simps[ID].hsuf[i].BRemoved=true;
				simps.push_back(spx);
			}
		}
	}
}

void Cube::GetAll_C4simplex()
	/*Get all of the simplex with twice used hypersurface removed*/
{
  simps.clear();
	//assert( Nints>4);
    if( Nints <=4 ){
        std::cerr<<"#intersections less than 5 \n";
        exit( -1 );
    }
	C4simplex simp0=C4simplex(ints);
	simps.push_back(simp0);//The first simplex

	for(int i=5; i<Nints; i++){
		int Simps_size=simps.size();
		for(vector<C4simplex>::size_type j=0; j != Simps_size; ++j){
			ConsNew_4simplex(ints[i], j);//This step will change simps.size()
		}
		if(simps.size() > Simps_size){
			for(vector<C4simplex>::size_type j=Simps_size; j != simps.size(); ++j)
				for(vector<C4simplex>::size_type k=j+1; k != simps.size(); ++k){
					Remv_Inner_SF(j, k);
				}//Remove the inner surface between new produced simplex
		}
	}
}

/** Notice DA0, DA1, DA2, DA3 are d\Sigma^{\mu}, the final results should be DA0, -DA1, -DA2, -DA3 */
void Cube::CalcSF2(double &DA0, double& DA1, double& DA2, double &DA3, \
                   double &vx_sf,  double& vy_sf,  double& vz_sf, \
                   double &tau_sf, double& x_sf, double &y_sf, double & z_sf){
	CalcInts();
	CalcVelocity();

    tau_sf = Origin.t + DTD * MassCenter.t;
    x_sf   = Origin.x + DXD * MassCenter.x;
    y_sf   = Origin.y + DYD * MassCenter.y;
    z_sf   = Origin.z + DZD * MassCenter.z;

    vx_sf = V.x;
    vy_sf = V.y;
    vz_sf = V.z;

	DA0 = 0.0;
	DA1 = 0.0;
	DA2 = 0.0;
	DA3 = 0.0;
	Point v1,v2,v3; /*v1,v2,v3 3 vectors that span one facet of the hypersurfaces of 4-simplex in 4D space.*/
	Point VHyperSurf;
	const double tau= Origin.t + DTD*MassCenter.t;
	const double coef=1.0/6.0;

	if(Nints==4){//if there are only 4 intersections, we can calc hypersurf directly
		v1=ints[1].pos-ints[0].pos;
		v2=ints[2].pos-ints[0].pos;
		v3=ints[3].pos-ints[0].pos;
		double ds0=  ( v1.x*(v2.y*v3.z-v2.z*v3.y) + v1.y*(v2.z*v3.x-v2.x*v3.z) + v1.z*(v2.x*v3.y-v2.y*v3.x) );
		double ds1= -( v1.t*(v2.y*v3.z-v2.z*v3.y) + v1.y*(v2.z*v3.t-v2.t*v3.z) + v1.z*(v2.t*v3.y-v2.y*v3.t) );
		double ds2=  ( v1.t*(v2.x*v3.z-v2.z*v3.x) + v1.x*(v2.z*v3.t-v2.t*v3.z) + v1.z*(v2.t*v3.x-v2.x*v3.t) );
		double ds3= -( v1.t*(v2.x*v3.y-v2.y*v3.x) + v1.x*(v2.y*v3.t-v2.t*v3.y) + v1.y*(v2.t*v3.x-v2.x*v3.t) );
		double sign = (ds0*Vsurf.t + ds1*Vsurf.x +ds2*Vsurf.y +ds3*Vsurf.z >0) ? 1.0 : -1.0;
		DA0 +=sign*tau*DXD*DYD*DZD*coef* ds0;
		DA1 +=sign*tau*DTD*DYD*DZD*coef* ds1;
		DA2 +=sign*tau*DTD*DXD*DZD*coef* ds2;
		DA3 +=sign*DTD*DXD*DYD*coef* ds3;
	}//Note here ds0=ds^0, ds1=ds_1, ds2=ds_2, ds3=ds_3. 

	else if(Nints>4){
		/*if there are more than 4 intersections, we can calc the convex hull of the 4D hyper-volum,
		  and choose the hypersurface by Angle(v_i . Vsurf)<\pi/2. where v_i is the normal vector of
		  the 4D hypersurface */
		GetAll_C4simplex();
		//cout<<"The number of simplex is "<<simps.size()<<endl;

		for(std::vector<C4simplex>::size_type i=0; i != simps.size(); ++i ){
			/*For all the hypersurface on the convex hull, projecting them on the flow direction*/
			for(int j=0; j<5; j++){
				if(simps[i].hsuf[j].BRemoved == false){
				  Point v=simps[i].hsuf[j].Vhs;   // v is the normal vector of the hypersurface
				  //Vsurf is toward low energy density direction.
				  double DNV= v.t*Vsurf.t + v.x*Vsurf.x + v.y*Vsurf.y + v.z*Vsurf.z ; 
					if(DNV>0){
						DA0 += tau*DXD*DYD*DZD*coef*v.t;
						DA1 += tau*DTD*DYD*DZD*coef*v.x; 
						DA2 += tau*DTD*DXD*DZD*coef*v.y; 
						DA3 += DTD*DXD*DYD*coef*v.z;
					}
				}
			}//end for j
		}//end for i
	}//end elseif
}




void Cube::SetOrigin(CD &t, CD & x, CD & y, CD & z)
{
	Origin.t = t;
	Origin.x = x;
	Origin.y = y;
	Origin.z = z;
}

Cube::Cube(double IValue, double dtd, double dxd, double dyd, double dzd):IntsValue(IValue),DTD(dtd),DXD(dxd),DYD(dyd),DZD(dzd){
	MassCenter= Point(0.0,0.0,0.0,0.0);
	Origin = Point(0.0, 0.0, 0.0, 0.0);
	Vsurf  = Point(0.0, 0.0, 0.0, 0.0);
	Nints= 0;
	ints = new CIntersection[32];
	for(int n=0; n<=1; n++)
		for(int i=0; i<=1; i++)
			for(int j=0; j<=1; j++)
				for(int k=0; k<=1; k++){
					int index=8*n+4*i+2*j+k;
					Point pt(n,i,j,k);
					corners[index].id  = index;
					corners[index].pos = pt;
					corners[index].value = 0.0;
				}//end for n i j k  initialize the corners

	int countEdge=0;
	for(int i1=0; i1<15; i1++)
		for(int j1=i1+1; j1<16; j1++)
		{
			if(IsEdge(corners[i1],corners[j1])){
				edges[countEdge].id=countEdge;
				edges[countEdge].lc=corners[i1];
				edges[countEdge].rc=corners[j1];
				countEdge++ ;
			}
		}// construct edges
}

/*****clac V throuth 4D interpolation ****************/
void Cube::CalcVelocity(){
	double t=MassCenter.t;
	double x=MassCenter.x;
	double y=MassCenter.y;
	double z=MassCenter.z;
	double A, B, C, D;
	V=Point(0.0,0.0,0.0,0.0);
	for(int m=0; m!=10; m++)pi[m] = 0.0;

	for(int n=0; n!=2; ++n)
		for(int i=0; i!=2; ++i)
			for(int j=0; j!=2; ++j)
				for(int k=0; k!=2; ++k)
				{
					if(n==0)A=1.0-t;else A=t;
					if(i==0)B=1.0-x;else B=x;
					if(j==0)C=1.0-y;else C=y;
					if(k==0)D=1.0-z;else D=z;
					int ID=8*n+4*i+2*j+k;
					V.x += A*B*C*D*corners[ID].VL[0];
					V.y += A*B*C*D*corners[ID].VL[1];
					V.z += A*B*C*D*corners[ID].VL[2];
					for(int m=0; m!=10; ++m)
					  pi[m] += A*B*C*D*corners[ID].pi[m];
				  }
	V.t= 1.0;
}

//#define CUBE_TEST

#ifdef CUBE_TEST
#include<iomanip>
using namespace std;
int main(int argc, char **argv)
{
	double ED[16];
	double VL[16][3];
	for(int i=0; i<8; i++){
		ED[i]=2;
		ED[8+i]=3;

		VL[2*i][0]=2;
		VL[2*i+1][0]=3;

		VL[2*i][1]=2;
		VL[2*i+1][1]=3;

		VL[2*i][2]=2;
		VL[2*i+1][2]=3;
	}

    /**  Edec, DTD, DXD, DYD, DZD */
	Cube cube1(2.5,1.0,1.0,1.0,1.0);
	cube1.SetOrigin( 0, 0, 0, 0 );
	cube1.SetCornersValue(ED);
	cube1.SetCornersVelocity(VL);

	double da0, da1, da2, da3, t, x, y, z, vx, vy, vz;
	cube1.CalcSF2(da0,da1,da2,da3, vx, vy, vz, t, x, y, z );

	cout<<"DA0="<<da0<<" DA1="<<-da1<<" DA2="<<-da2<<" DA3="<<-da3\
        <<"x  ="<<x  <<" y  ="<< y  <<" z  ="<<z   <<" t=  "<<t   \
        <<"vx ="<<vx <<"vy  ="<<vy  <<"vz  ="<<vz<<endl;
	//cout<<"Direction of surface"<<setw(5)<<cube1.Vsurf.t<<setw(5)<<cube1.Vsurf.x<<setw(5)<<cube1.Vsurf.y<<setw(5)<<cube1.Vsurf.z<<endl;


	return 0;
}
#endif
