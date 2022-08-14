{
    volScalarField& he = thermo.he();

    // correct alphat
    alphat = rho*turbulence->nut()/Prt;
    alphat.correctBoundaryConditions();

    fvScalarMatrix EEqn
    (
        fvm::ddt(rho, he) + fvm::div(phi, he)
      + fvc::ddt(rho, K) + fvc::div(phi, K)
      - dpdt
      - fvm::laplacian(thermo.alphaEff(alphat), he)
      + diffAlphaD
     ==
        fvc::div(hDiffCorrFlux)
      + fvModels.source(rho, he)
    );

    EEqn.relax();

    EEqn.solve();
}
