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
    Foam::CanteraSpecies

Description
    Foam::CanteraSpecies

SourceFiles
    CanteraSpecies.C

\*---------------------------------------------------------------------------*/

#ifndef CanteraSpecies_H
#define CanteraSpecies_H


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

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                         Class CanteraSpecies Declaration
\*---------------------------------------------------------------------------*/

class CanteraSpecies
{
protected:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
    IOdictionary CanteraTorchProperties_;
    const word CanteraMechanismFile_;
    std::shared_ptr<Cantera::Solution> CanteraSolution_;
    std::shared_ptr<Cantera::ThermoPhase> CanteraGas_;
    word transportModelName_;
    std::shared_ptr<Cantera::Transport> CanteraTransport_;
    hashedWordList species_;
    PtrList<volScalarField> Y_;

public:
    static word typeName()
    {
        return "CanteraSpecies";
    }

    ~CanteraSpecies();

    //- Construct from dictionary, mesh and phase name
    CanteraSpecies(const dictionary&, const fvMesh&, const word&);

    void calcCp(const scalar& p, const scalar& T) const;

    void calcMu(const scalar& p, const scalar& T) const;

    void calcH(const scalar& p, const scalar& T) const;


    scalar Cp(label i, scalar p, scalar T) const;

    scalar mu(label i, scalar p, scalar T) const;

    scalar Ha(label i, scalar p, scalar T) const;

    scalar Hf(label i) const
    {
        return CanteraGas_->Hf298SS(i)/CanteraGas_->molecularWeight(i); // J/kg
    }

    scalar Hs(label i, scalar p, scalar T) const
    {
        return Ha(i, p, T) - Hf(i); // J/kg
    }

    scalar Wi(label i) const
    {
        return CanteraGas_->molecularWeight(i); // kg/kmol
    }


    scalar kappa(label i, scalar p, scalar T) const
    {
        // should be kappa of single species
        // but now lack of access function in Cantera
        return CanteraTransport_->thermalConductivity();
    } // W/m/K

    const hashedWordList& species() const {return species_;}

    size_t nSpecies() {return CanteraGas_->nSpecies();}

    std::shared_ptr<Cantera::ThermoPhase> CanteraGas() {return CanteraGas_;}

    std::shared_ptr<Cantera::Solution> CanteraSolution() {return CanteraSolution_;}

    std::shared_ptr<Cantera::Transport> CanteraTransport() {return CanteraTransport_;}

    const word& transportModelName() {return transportModelName_;}


    PtrList<volScalarField>& Y() {return Y_;}
    const PtrList<volScalarField>& Y() const {return Y_;}

    volScalarField& Y(const label i) {return Y_[i];}
    const volScalarField& Y(const label i) const {return Y_[i];}

    volScalarField& Y(const word& specieName) {return Y_[species_[specieName]];}
    const volScalarField& Y(const word& specieName) const {return Y_[species_[specieName]];}

    bool contains(const word& specieName) const {return species_.found(specieName);}

    bool solve(label speciei) const {
        return true;//return speciei != defaultSpecieIndex_ && active_[speciei];
    }

    label index(const volScalarField& Yi) const
    {
        return species_[Yi.name()];
    }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
