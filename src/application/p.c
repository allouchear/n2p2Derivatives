double*** d3f getNumericThird(Prediction& prediction,double step)
{
	Structure& s = prediction.structure;
	int nAtoms = s.numAtoms;
	vector<double> v(nAtoms);
	vector< vector<double> > gp(3,v);
	vector< vector<double> > gm(3,v);


	double*** d3f = new3Dtable(3*nAtoms);
	double stepConv = prediction.getConvLength()*step;

	for(int i=0;i<nAtoms;i++)
	for(int k=0;k<3;k++)
	{
		int id=3*i+k;
    		prediction.readStructureFromFile("input.data");
		// readStructureFromFile : in conv units
		s.atoms[i].r[k] += stepConv;
		// predict hessian in physical units
		auto gp = prediction.getHighDerivatives(2);

    		prediction.readStructureFromFile("input.data");
		s.atoms[i].r[k] -= stepConv;
		auto gm = prediction.getHighDerivatives(2);

		for(int j=0;j<nAtoms3;j++)
		for(int k=0;k<nAtoms3;k++)
		{
				// g are in phys units, so use pysunit for step
			d3f[id][j][k] =  (gp[j][k]-gm[j][k])/step/2;
		}
	}
	return d3f;
}
