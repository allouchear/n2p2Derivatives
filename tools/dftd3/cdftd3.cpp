#include <iostream>
#include <cstring>
#include <cstdio>
#include "AtomsProp.h"

using namespace std;

extern"C" {
void  dftd3c_(char* func, int* version, int* nAtoms, int* atnum, double* coords, double* edisp, double* grads);
void  dftd3cpbc_(char* func, int* version, int* nAtoms, int* atnum, double* coords, double* latVecs, double* edisp, double* grads, double* stress);
}
struct Atom
{
	double coordinates[3];
	double forces[3];
	double charge; // partial charge
	char symbol[10];
};
struct Molecule
{
	int nAtoms;
	Atom* atoms;
	double energy;
	double charge;
	double dipole[3];
        int nlattice;
	double lattice[3][3];
};

#define BSIZE 1024

/********************************************************************************/
static void freeMolecule(Molecule* mol)
{
	if(mol)
	{
		if(mol->atoms) delete [] mol->atoms;
		delete mol;
	}
}
/********************************************************************************/
static Molecule* readOneMolecule(FILE* file)
{
	char tmp[BSIZE];
	char symbol[BSIZE];
	char dum[BSIZE];
	double x,y,z;
	double fx,fy,fz;
	int nAtoms;
	Molecule* mol = NULL;
	double energy;
	long int geompos = -1;
	double charge;
	double d;
	 
	// got begin
	while(!feof(file))
	{
		if(!fgets(tmp,BSIZE,file))break;
		if(strstr(tmp,"begin")) { 
			geompos = ftell(file);
			break;
		}
	}
	if(geompos<0) return NULL;
	// compute nAtoms
   	nAtoms = 0;
	while(!feof(file))
	{
		if(!fgets(tmp,BSIZE,file))break;
		if(strstr(tmp,"atom")) { 
			nAtoms++;
		}
		if(strstr(tmp,"end")) break; 
	}
	mol = new Molecule;
	mol->nAtoms = nAtoms;
	mol->atoms = new Atom [nAtoms];
	fseek(file, geompos, SEEK_SET);
	mol->energy = 0;
	mol->charge = 0;
	mol->dipole[0] = 0;
	mol->dipole[1] = 0;
	mol->dipole[2] = 0;
	mol->nlattice = 0;
	for(int i=0;i<3;i++) for(int j=0;j<3;j++) mol->lattice[i][j] = 0.0;
		
	int ia =0;
	while(!feof(file))
	{
		if(!fgets(tmp,BSIZE,file))break;
		if(strstr(tmp,"atom")) { 
			if(10==sscanf(tmp,"%s %lf %lf %lf %s %lf %lf %lf %lf %lf",dum, &x, &y, &z, symbol,&charge, &d, &fx, &fy, &fz)) { 
			sprintf(mol->atoms[ia].symbol,"%s",symbol);
			mol->atoms[ia].coordinates[0] = x;
			mol->atoms[ia].coordinates[1] = y;
			mol->atoms[ia].coordinates[2] = z;
			mol->atoms[ia].charge = charge;
			mol->atoms[ia].forces[0] = fx;
			mol->atoms[ia].forces[1] = fy;
			mol->atoms[ia].forces[2] = fz;
			ia++;
			}
		}
		if(strstr(tmp,"lattice")) { 
			if(4==sscanf(tmp,"%s %lf %lf %lf",dum, &x, &y, &z)) { 
			mol->lattice[mol->nlattice][0] = x;
			mol->lattice[mol->nlattice][1] = y;
			mol->lattice[mol->nlattice][2] = z;
			mol->nlattice++;
			}
		}
		if(strstr(tmp,"energy")) sscanf(tmp,"%s %lf",dum, &mol->energy);
		if(strstr(tmp,"charge")) sscanf(tmp,"%s %lf",dum, &mol->charge);
		if(strstr(tmp,"dipole")) sscanf(tmp,"%s %lf %lf %lf",dum, &mol->dipole[0], &mol->dipole[1], &mol->dipole[2]);
		if(strstr(tmp,"end")) break; 
	}
	if(ia != nAtoms) 
	{
		fprintf(stderr,"Error : ia != nAtoms in readMolecule\n");
	}
	return mol;
}
/********************************************************************************/
static void saveMolecule(Molecule* mol, FILE* file)
{
	
	fprintf(file,"begin\n");
	for(int i=0;i<mol->nlattice;i++)
	{
		fprintf(file,"%s %0.14lf %0.14lf %0.14lf\n","lattice", 
			mol->lattice[i][0],
			mol->lattice[i][1],
			mol->lattice[i][2]);
	}
	for(int ia=0;ia<mol->nAtoms;ia++)
	{
		fprintf(file,"%s %0.14lf %0.14lf %0.14lf %s %0.14lf %0.14lf %0.14lf %0.14lf %0.14lf\n","atom", 
			mol->atoms[ia].coordinates[0],
			mol->atoms[ia].coordinates[1],
			mol->atoms[ia].coordinates[2],
			mol->atoms[ia].symbol,
			mol->atoms[ia].charge,
			0.0,
			mol->atoms[ia].forces[0],
			mol->atoms[ia].forces[1],
			mol->atoms[ia].forces[2]
			);
	}
	fprintf(file,"energy %0.14lf\n", mol->energy);
	fprintf(file,"charge %0.14lf\n", mol->charge);
	fprintf(file,"dipole %0.14lf %0.14lf %0.14lf\n", mol->dipole[0],  mol->dipole[1],  mol->dipole[2]);
	fprintf(file,"end\n");
}

/********************************************************************************/
static void computeDFTD3(Molecule* mol, FILE* fileoutPlus, FILE* fileoutMinus)
{
   double edisp = 0.0;
   int nAtoms;
   int* atnum;
   double* grads;
   double stress[9];
   double latVecs[9];
   double* coords;
   int version = 4;

   nAtoms = mol->nAtoms;
   atnum = new int [nAtoms];
   for(int ia=0;ia<mol->nAtoms;ia++)
	atnum[ia] = (int)getAtomicNumberFromSymbol(mol->atoms[ia].symbol);
   coords = new double [3*nAtoms];
   grads = new double [3*nAtoms];
   int k = 0;
   for(int i=0;i<3;i++)
   for(int ia=0;ia<mol->nAtoms;ia++)
   {
      coords [k] = mol->atoms[ia].coordinates[i]; 
      k++;
    }
   if(mol->nlattice!=3)
   {
      cout<<"call dftd3c_ "<<endl;
      dftd3c_((char*)"dftb3",&version, &nAtoms, atnum, coords, &edisp, grads);
   }
   else
   {
      int k = 0;
      for(int i=0;i<mol->nlattice;i++)
      for(int j=0;j<3;j++)
      {
          latVecs [k] = mol->lattice[i][j]; 
          k++;
      }
      cout<<"call dftd3cpbc_ "<<endl;
      dftd3cpbc_((char*)"dftb3",&version, &nAtoms, atnum, coords, latVecs, &edisp, grads, stress);
   }
   //dftd3cpbc_(&version,&nAtoms, atnum, coords, latVecs, &edisp, grads, stress);
   cout<<"edisp = "<<edisp<<endl;
   k = 0;
   for(int i=0;i<3;i++)
   for(int ia=0;ia<mol->nAtoms;ia++)
   {
      mol->atoms[ia].forces[i] += grads[k]; 
      k++;
    }
    mol->energy -= edisp;
    saveMolecule(mol,fileoutMinus);
   k = 0;
   for(int i=0;i<3;i++)
   for(int ia=0;ia<mol->nAtoms;ia++)
   {
      mol->atoms[ia].forces[i] -= 2*grads[k]; 
      k++;
    }
    mol->energy += 2*edisp;
    saveMolecule(mol,fileoutPlus);

}

int main()
{

   defineDefaultAtomsProp();
   FILE* file = fopen("input.data","r");
   if(!file)
   {
	fprintf(stderr,"I cannot open input.data file\n");
	return 1;
   }
   char* cfileoutPlus=(char*) "inputPlus.data";
   FILE* fileoutPlus=fopen(cfileoutPlus,"w");
   char* cfileoutMinus=(char*)"inputMinus.data";
   FILE* fileoutMinus=fopen(cfileoutMinus,"w");

   Molecule* mol;
   while(true){
	mol = readOneMolecule(file);
   	if(mol)
   	{
		//saveMolecule(mol,stdout);
		computeDFTD3(mol,  fileoutPlus,  fileoutMinus);
		freeMolecule(mol);
   	}
	else break;
    }
    fprintf(stderr,"see %s and %s files\n",  cfileoutPlus, cfileoutMinus);


   return 0;
}
