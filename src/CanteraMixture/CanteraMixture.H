/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2019 OpenFOAM Foundation
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

Class
    Foam::CanteraMixture

Description
    Foam::CanteraMixture

SourceFiles
    CanteraMixture.C

\*---------------------------------------------------------------------------*/

#ifndef CanteraMixture_H
#define CanteraMixture_H


#include "cantera/zerodim.h"
#include "cantera/transport.h"
#include "cantera/thermo/IdealGasPhase.h"
#include "IOdictionary.H"
#include "fvMesh.H"
#include "word.H"
#include "volMesh.H"
#include "GeometricField.H"
#include "volFieldsFwd.H"
#include "fvPatchField.H"
#include "hashedWordList.H"
#include "physicoChemicalConstants.H"
#include "CanteraSpecies.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                         Class CanteraMixture Declaration
\*---------------------------------------------------------------------------*/

class CanteraMixture : public CanteraSpecies
{
    const volScalarField& Tref_;
    const volScalarField& pref_;
    mutable scalarList yTemp_;


    //- Construct as copy (not implemented)
    CanteraMixture(const CanteraMixture&);


public:

    typedef CanteraMixture basicMixtureType;
    typedef CanteraMixture thermoType;
    typedef CanteraMixture thermoMixtureType;
    typedef CanteraMixture transportMixtureType;
    static const bool incompressible = false;
    static const bool isochoric = false;


    //- Construct from dictionary, mesh and phase name
    CanteraMixture(const dictionary&, const fvMesh&, const word&);



    //- Return the instantiated type name
    static word typeName()
    {
        return "CanteraMixture";
    }

    const CanteraMixture& cellThermoMixture(const label celli) const;

    const CanteraMixture& patchFaceThermoMixture(const label patchi, const label facei) const;

    const transportMixtureType& cellTransportMixture
    (
        const label celli
    ) const
    {
        return cellThermoMixture(celli);
    }

    const transportMixtureType& patchFaceTransportMixture
    (
        const label patchi,
        const label facei
    ) const
    {
        return patchFaceThermoMixture(patchi, facei);
    }

    const transportMixtureType& cellTransportMixture
    (
        const label,
        const thermoMixtureType& thermoMixture
    ) const
    {
        return thermoMixture;
    }

    const transportMixtureType& patchFaceTransportMixture
    (
        const label,
        const label,
        const thermoMixtureType& thermoMixture
    ) const
    {
        return thermoMixture;
    }

    scalar THE
    (
        const scalar& h,
        const scalar& p,
        const scalar& T
    ) const;

    scalar psi
    (
        const scalar& p,
        const scalar& T
    ) const
    {
        return CanteraGas_->meanMolecularWeight()/CanteraGas_->RT();
    }

    scalar mu
    (
        const scalar& p,
        const scalar& T
    ) const
    {
        return CanteraTransport_->viscosity();
    }

    scalar kappa
    (
        const scalar& p,
        const scalar& T
    ) const
    {
        return CanteraTransport_->thermalConductivity(); // W/m/K
    }

    scalar HE
    (
        const scalar& p,
        const scalar& T
    ) const
    {
        return CanteraGas_->enthalpy_mass(); // J/kg
    }

    scalar Ha
    (
        const scalar& p,
        const scalar& T
    ) const
    {
        return CanteraGas_->enthalpy_mass(); // J/kg
    }

    scalar Hs
    (
        const scalar& p,
        const scalar& T
    ) const
    {
        return CanteraGas_->enthalpy_mass() - Hf(); // J/kg
    }

    scalar Hf() const; // J/kg

    scalar Cp
    (
        const scalar& p,
        const scalar& T
    ) const
    {
        return CanteraGas_->cp_mass(); // J/kg/K
    }

    scalar Cv
    (
        const scalar& p,
        const scalar& T
    ) const
    {
        return CanteraGas_->cv_mass(); // J/kg/K
    }

    scalar gamma
    (
        const scalar& p,
        const scalar& T
    ) const
    {
        return Cp(p, T)/Cv(p, T);
    }

    scalar Cpv
    (
        const scalar& p,
        const scalar& T
    ) const
    {
        return Cp(p, T);
    }

    scalar CpByCpv
    (
        const scalar& p,
        const scalar& T
    ) const
    {
        return 1;
    }

    scalar W() const
    {
        return CanteraGas_->meanMolecularWeight();
    }

    static inline word heName() {return "ha";}
    static inline bool enthalpy() {return true;}

    //- Read dictionary
    void read(const dictionary&);
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// #ifdef NoRepository
//     #include "CanteraMixture.C"
// #endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
