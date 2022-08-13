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

#include "CanteraSpecies.H"
#include "fvMesh.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

class Foam::CanteraSpecies::Impl
{
private:
    CanteraSpecies* CanteraSpeciesPtr;
    scalarList HaTemp_;
    scalarList CpTemp_;
    scalarList muTemp_;

public:
    Impl(/*CanteraSpecies* CanteraSpeciesPtr, */const dictionary& thermoDict, const fvMesh& mesh, const word& phaseName){}

    void calcCp(const scalar& p, const scalar& T)
    {
        // const scalar RR = constant::physicoChemical::R.value()*1e3; // J/(kmol·k)

        // CanteraSpeciesPtr->CanteraGas_->setState_TP(T, p);

        // scalarList Cp_R(CanteraSpeciesPtr->nSpecies());
        // CanteraSpeciesPtr->CanteraGas_->getCp_R(Cp_R.begin());
        // CpTemp_ = Cp_R*RR;
    }

    void calcMu(const scalar& p, const scalar& T)
    {
        // CanteraSpeciesPtr->CanteraGas_->setState_TP(T, p);

        // CanteraSpeciesPtr->CanteraTransport_->getSpeciesViscosities(muTemp_.begin());
    }

    void calcH(const scalar& p, const scalar& T)
    {
        // const scalar RT = constant::physicoChemical::R.value()*1e3*T; // J/kmol/K

        // CanteraSpeciesPtr->CanteraGas_->setState_TP(T, p);

        // scalarList Ha_RT(CanteraSpeciesPtr->nSpecies());
        // CanteraSpeciesPtr->CanteraGas_->getEnthalpy_RT(Ha_RT.begin());
        // HaTemp_ = Ha_RT*RT;
    }
};


Foam::CanteraSpecies::CanteraSpecies(const dictionary& thermoDict, const fvMesh& mesh, const word& phaseName)
:
    pimpl_(std::make_unique<Impl>(/*this, */thermoDict, mesh, phaseName)),
    CanteraTorchProperties_
    (
        IOobject
        (
            "CanteraTorchProperties",
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
        )
    ),
    CanteraMechanismFile_(CanteraTorchProperties_.lookup("CanteraMechanismFile")),
    CanteraSolution_(Cantera::newSolution(CanteraMechanismFile_, "")),
    CanteraGas_(CanteraSolution_->thermo()),
    transportModelName_(CanteraTorchProperties_.lookup("transportModel")),
    CanteraTransport_(newTransportMgr(transportModelName_, CanteraGas_.get())),
    Y_(nSpecies()),
    HaTemp_(nSpecies()),
    CpTemp_(nSpecies()),
    muTemp_(nSpecies())
{
   forAll(Y_, i)
    {
        species_.append(CanteraGas_->speciesName(i));
        Info<<"species "<<CanteraGas_->speciesName(i)<<" added!"<<endl;
    }

    tmp<volScalarField> tYdefault;

    forAll(Y_, i)
    {
        IOobject header
        (
            species_[i],
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ
        );

        // check if field exists and can be read
        if (header.headerOk())
        {
            Y_.set
            (
                i,
                new volScalarField
                (
                    IOobject
                    (
                        species_[i],
                        mesh.time().timeName(),
                        mesh,
                        IOobject::MUST_READ,
                        IOobject::AUTO_WRITE
                    ),
                    mesh
                )
            );
        }
        else
        {
            // Read Ydefault if not already read
            if (!tYdefault.valid())
            {
                word YdefaultName("Ydefault");

                IOobject timeIO
                (
                    YdefaultName,
                    mesh.time().timeName(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                );

                IOobject constantIO
                (
                    YdefaultName,
                    mesh.time().constant(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                );

                IOobject time0IO
                (
                    YdefaultName,
                    Time::timeName(0),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                );

                if (timeIO.headerOk())
                {
                    tYdefault = new volScalarField(timeIO, mesh);
                }
                else if (constantIO.headerOk())
                {
                    tYdefault = new volScalarField(constantIO, mesh);
                }
                else
                {
                    tYdefault = new volScalarField(time0IO, mesh);
                }
            }

            Y_.set
            (
                i,
                new volScalarField
                (
                    IOobject
                    (
                        species_[i],
                        mesh.time().timeName(),
                        mesh,
                        IOobject::NO_READ,
                        IOobject::AUTO_WRITE
                    ),
                    tYdefault()
                )
            );
        }
    }
}




Foam::CanteraSpecies::~CanteraSpecies(){}



    void Foam::CanteraSpecies::calcCp(const scalar& p, const scalar& T) const
    {
        pimpl_->calcCp(p, T);
    }

    void Foam::CanteraSpecies::calcMu(const scalar& p, const scalar& T) const
    {
        pimpl_->calcMu(p, T);
    }

    void Foam::CanteraSpecies::calcH(const scalar& p, const scalar& T) const
    {
        pimpl_->calcH(p, T);
    }


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //


// ************************************************************************* //
