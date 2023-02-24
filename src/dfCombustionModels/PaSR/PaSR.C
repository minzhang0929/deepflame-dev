/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "PaSR.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ReactionThermo>
Foam::combustionModels::PaSR<ReactionThermo>::PaSR
(
    const word& modelType,
    ReactionThermo& thermo,
    const compressibleTurbulenceModel& turb,
    const word& combustionProperties
)
:
    laminar<ReactionThermo>(modelType, thermo, turb, combustionProperties),
    mixingScaleDict_(this->coeffs().subDict("mixingScale")),
    chemistryScaleDict_(this->coeffs().subDict("chemistryScale")),
    mixingScaleType_(mixingScaleDict_.lookup("type")),
    chemistryScaleType_(chemistryScaleDict_.lookup("type")),
    mixingScaleCoeffs_(mixingScaleDict_.optionalSubDict(mixingScaleType_ + "Coeffs")),
    chemistryScaleCoeffs_(chemistryScaleDict_.optionalSubDict(chemistryScaleType_ + "Coeffs")),
    fuel_(chemistryScaleCoeffs_.lookup("fuel")),
    oxidizer_(chemistryScaleCoeffs_.lookup("oxidizer")),
    tmix_
    (
        IOobject
        (
            "tmix",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar(dimTime, 0)
    ),
    tc_
    (
        IOobject
        (
            "tmix",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar(dimTime, GREAT)
    ),
    Da_
    (
        IOobject
        (
            "Da",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar(dimless, 0)
    ),
    Z_
    (
        IOobject
        (
            "Z",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar(dimless, 0)
    ),
    Zvar_
    (
        IOobject
        (
            "Zvar",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar(dimless, 0)
    ),
    Chi_
    (
        IOobject
        (
            "Chi",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar(dimless/dimTime, SMALL)
    ),    
    kappa_
    (
        IOobject
        (
            thermo.phasePropertyName(typeName + ":kappa"),
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar(dimless, 1.0)
    ),
    transportChi_(mixingScaleCoeffs_.lookupOrDefault("transportChi",false))
{
     Cmix_=mixingScaleCoeffs_.lookupOrDefault("Cmix",0.1);

     //- adopted from Ferrarotti et al. 2019 PCI
     Cd1_=mixingScaleCoeffs_.lookupOrDefault("Cd1",1.1);
     Cd2_=mixingScaleCoeffs_.lookupOrDefault("Cd2",0.8);
     Cp1_=mixingScaleCoeffs_.lookupOrDefault("Cp1",0.9);
     Cp2_=mixingScaleCoeffs_.lookupOrDefault("Cp2",0.72);

     fields_.add(Z_);
     fields_.add(Zvar_);
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class ReactionThermo>
Foam::combustionModels::PaSR<ReactionThermo>::~PaSR()
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

template<class ReactionThermo>
void Foam::combustionModels::PaSR<ReactionThermo>::correct()
{
    laminar<ReactionThermo>::correct();
 
    tmp<volScalarField> tk(this->turbulence().k());
    const volScalarField& k = tk();

    tmp<volScalarField> tepsilon(this->turbulence().epsilon());
    const volScalarField& epsilon = tepsilon();

    tmp<volScalarField> tmu(this->turbulence().mu());
    const volScalarField& mu = tmu();

    tmp<volScalarField> trho(this->rho());
    const volScalarField& rho = trho();

    dimensionedScalar smallEpsilon("smallEpsilon",dimensionSet(0, 2, -3, 0, 0, 0, 0), SMALL);	
    dimensionedScalar zeroTimeSquare("zeroTimeSquare",dimensionSet(0, 0, 2, 0, 0, 0, 0), 0.0);
    dimensionedScalar zeroTime("zeroTime",dimensionSet(0, 0, 1, 0, 0, 0, 0), 0.0);
    //-mixing time scale 
    if(mixingScaleType_=="globalScale")
    {
       tmix_=Cmix_*k/(epsilon+smallEpsilon);
    }

    if(mixingScaleType_=="kolmogorovScale")
    {
      // tmix_=sqrt(mag(mu/rho/(epsilon+smallEpsilon)));  
      tmix_=sqrt(max(mu/rho/(epsilon+smallEpsilon), zeroTimeSquare));  
    }

    if(mixingScaleType_=="geometriMeanScale")
    {
       tmix_=sqrt((max(k/(epsilon+smallEpsilon),zeroTime))*sqrt(max(mu/rho/(epsilon+smallEpsilon),zeroTimeSquare)));
    }

    if(mixingScaleType_=="dynamicScale")
    {
        transport();
        tmix_=Zvar_/Chi_;
    }


    //- chemitry time scale
    if(chemistryScaleType_=="globalConvertion")
    {



       const label specieFuel= this->chemistryPtr_->species()[fuel_];
	   const label specieOxidizer = this->chemistryPtr_->species()[oxidizer_];

       const volScalarField& Yfuel = this->chemistryPtr_->Y()[specieFuel];
       const volScalarField& Yoxidizer = this->chemistryPtr_->Y()[specieOxidizer];


     //- initialize fuel and oxidizer chemitry time scale
        volScalarField t_fuel=tc_;
		volScalarField t_oxidizer=tc_;

	   forAll(rho,cellI)
	   {

		scalar RR_fuel = this->chemistryPtr_->RR(specieFuel)[cellI];
		scalar RR_oxidizer = this->chemistryPtr_->RR(specieOxidizer)[cellI];

		if( (RR_oxidizer < 0.0)  &&  (Yoxidizer[cellI] > 1e-10) )							
		{			
			t_oxidizer[cellI] =  -rho[cellI] * Yoxidizer[cellI]/(RR_oxidizer);   
		}
	 
		if	( (RR_fuel < 0.0) && (Yfuel[cellI] > 1e-10))
		{								
			t_fuel[cellI] =  -rho[cellI] * Yfuel[cellI]/(RR_fuel);   
		}

		tc_[cellI] = min(t_oxidizer[cellI],t_fuel[cellI]);
		
	  }

    }

    if(chemistryScaleType_=="formationRate")
    {
       tc_ = this->tc();
    }


    forAll(kappa_, cellI)
    {
		kappa_[cellI] = (tmix_[cellI] > SMALL && tc_[cellI] > SMALL) ?  tc_[cellI]/(tc_[cellI] + tmix_[cellI]) : 1.0;
    }
}


template<class ReactionThermo>
Foam::tmp<Foam::fvScalarMatrix>
Foam::combustionModels::PaSR<ReactionThermo>::R(volScalarField& Y) const
{
    return kappa_*laminar<ReactionThermo>::R(Y);
}

template<class ReactionThermo>
bool Foam::combustionModels::PaSR<ReactionThermo>::read()
{
    if (laminar<ReactionThermo>::read())
    {
        this->coeffs().lookup("Cmix") >> Cmix_;
        return true;
    }
    else
    {
        return false;
    }
}


template<class ReactionThermo>
Foam::tmp<Foam::volScalarField>
Foam::combustionModels::PaSR<ReactionThermo>::Qdot() const
{
    return volScalarField::New
    (
        this->thermo().phasePropertyName(typeName + ":Qdot"),
        kappa_*laminar<ReactionThermo>::Qdot()
    );
}

template<class ReactionThermo>
void Foam::combustionModels::PaSR<ReactionThermo>::transport() 
{

    tmp<volScalarField> tmuEff(this->turbulence().muEff());
    const volScalarField& muEff = tmuEff();

    const surfaceScalarField& phi_ = this->mesh().objectRegistry::lookupObject<surfaceScalarField>("phi");
    const volVectorField& U_=this->mesh().objectRegistry::lookupObject<volVectorField>("U");

    tmp<fv::convectionScheme<scalar>> mvConvection
       (
          fv::convectionScheme<scalar>::New
           (
              this->mesh(),
              fields_,
              phi_,
              this->mesh().divScheme("div(phi,Z)")
           )
       );

  //- scalar dissipation rate equation 

  dimensionedScalar smallK_("smallK",dimVelocity*dimVelocity,SMALL);

  if(!transportChi_)
  {
      Chi_ = 2*this->turbulence().epsilon()/max(this->turbulence().k(),smallK_)*Zvar_; 
  }
  else   
  {
        scalar Sct=0.7;
    	volScalarField D1 = Cd1_*this->rho()*sqr(Chi_)/(Zvar_+SMALL);
        volScalarField D2 = Cd2_*this->rho()*this->turbulence().epsilon()/(this->turbulence().k()+smallK_)*Chi_;
        volScalarField P1 = 2.00*Cp1_*this->turbulence().epsilon()/(this->turbulence().k()+smallK_)*this->turbulence().mut()/Sct*magSqr(fvc::grad(Z_));
        volScalarField P2 = Cp2_*this->turbulence().mut()*Chi_/(this->turbulence().k()+smallK_)*(fvc::grad(U_) && dev(twoSymm(fvc::grad(U_))));

        volScalarField S_chi = P1 + P2 - D1 - D2;	

        fvScalarMatrix ChiEqn
        (
            fvm::ddt(this->rho(), Chi_)
            + fvm::div(phi_, Chi_)
            - fvm::laplacian(this->thermo().alpha()+this->turbulence().mut()/Sct, Chi_)
            ==
            S_chi
        );


        ChiEqn.relax();
        ChiEqn.solve();
        Chi_.max(0.00000001);
      //  Chi.min(maxChi);
  }


  //- mixture fraction  equation 
  fvScalarMatrix ZEqn
  (
     fvm::ddt(this->rho(), this->Z_)
    //+ fvm::div(this->phi(), this->Z_)
    + mvConvection->fvmDiv(phi_, this->Z_)
    - fvm::laplacian(muEff, this->Z_)
  );
  ZEqn.relax();
  ZEqn.solve("Z");
  this->Z_.max(0.0);

  //- mixtrue fraction variance equation
  
    fvScalarMatrix ZvarEqn
    (
        fvm::ddt(this->rho(), this->Zvar_)
      + mvConvection->fvmDiv(phi_, this->Zvar_)
      - fvm::laplacian(muEff, this->Zvar_)
      ==
      + 2*muEff*magSqr(fvc::grad(this->Z_))
      - this->rho()*Chi_
    );

    ZvarEqn.relax();
    ZvarEqn.solve("Zvar");
    this->Zvar_.max(0.0);
}
// ************************************************************************* //
