#define PERIODIC_TABLE_N_ROWS 10
#define PERIODIC_TABLE_N_COLUMNS 18


#define MAXISOTOP 10
struct SAtomsProp
{
	char *name;
	char *symbol;
	int atomicNumber;
	double covalentRadii;
	double bondOrderRadii;
	double vanDerWaalsRadii;
	double radii;
	int maximumBondValence;
	double mass;
	double electronegativity;
	int color[3];
	int nIsotopes;
	int iMass[MAXISOTOP];
	double rMass[MAXISOTOP];
	double abundances[MAXISOTOP];
};


char*** getPeriodicTable();
char* getSymbolUsingZ(int z);
double getAtomicNumberFromSymbol(char* symbol);
double getMasseFromSymbol(char* symbol);
bool testAtomDefine(char *Symb);
void propAtomFree(SAtomsProp* prop);
SAtomsProp propAtomGet(const char *);
void defineDefaultAtomsProp();
char *symbAtomGet(int); 


