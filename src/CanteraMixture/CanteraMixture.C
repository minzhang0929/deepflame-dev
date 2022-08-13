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

#include "CanteraMixture.H"
#include "fvMesh.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::CanteraMixture::CanteraMixture
(
    const dictionary& thermoDict,
    const fvMesh& mesh,
    const word& phaseName
)
:
    CanteraSpecies(thermoDict, mesh, phaseName),
    Tref_(mesh.objectRegistry::lookupObject<volScalarField>("T")),
    pref_(mesh.objectRegistry::lookupObject<volScalarField>("p")),
    yTemp_(nSpecies())
{
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::CanteraMixture::read(const dictionary& thermoDict)
{
}


const Foam::CanteraMixture& Foam::CanteraMixture::cellThermoMixture(const label celli) const
{
    forAll(Y_, i)
    {
        yTemp_[i] = Y_[i][celli];
    }
    CanteraGas_->setState_TPY(Tref_[celli], pref_[celli], yTemp_.begin());

    return *this;
}


const Foam::CanteraMixture& Foam::CanteraMixture::patchFaceThermoMixture
(
    const label patchi,
    const label facei
) const
{
    forAll(Y_, i)
    {
        yTemp_[i] = Y_[i].boundaryField()[patchi][facei];
    }
    CanteraGas_->setState_TPY(Tref_.boundaryField()[patchi][facei],pref_.boundaryField()[patchi][facei],yTemp_.begin());

    return *this;
}


Foam::scalar Foam::CanteraMixture::THE
(
    const scalar& h,
    const scalar& p,
    const scalar& T
) const
{
    CanteraGas_->setState_HP(h, p);
    return CanteraGas_->temperature();
}

Foam::scalar Foam::CanteraMixture::Hf() const
{
    scalar chemicalEnthalpy = 0;
    forAll(yTemp_, i)
    {
        chemicalEnthalpy += yTemp_[i]*CanteraGas_->Hf298SS(i)/CanteraGas_->molecularWeight(i);
    }
    return chemicalEnthalpy;
}

// ************************************************************************* //
