#ifndef GITRM_SURFACE_MODEL_HPP
#define GITRM_SURFACE_MODEL_HPP

#include "pumipic_adjacency.hpp"
#include "GitrmParticles.hpp" 
#include "GitrmMesh.hpp"

namespace o = Omega_h;
namespace p = pumipic;

OMEGA_H_DEVICE o::Real screeningLength(const o::Real projectileZ, 
    const o::Real targetZ) {
  o::Real bohrRadius = 5.29177e-11; //TODO
  return 0.885341*bohrRadius*pow(pow(projectileZ,(2.0/3.0)) + 
      pow(targetZ,(2.0/3.0)),(-1.0/2.0));
}

OMEGA_H_DEVICE o::Real stoppingPower (const o::Vector<3>& vel, const o::Real targetM, 
  const o::Real targetZ, const o::Real screenLength) {
  o::Real elCharge = ELECTRON_CHARGE;
  o::Real ke2 = 14.4e-10; //TODO
  o::Real amu = PTCL_AMU;
  o::Real atomZ = PARTICLE_Z;
  auto protonMass = PROTON_MASS;
  o::Real E0 = 0.5*amu*protonMass *1.0/elCharge * p::osh_dot(vel, vel);
  o::Real reducedEnergy = E0*(targetM/(amu+targetM))* (screenLength/(atomZ*targetZ*ke2));
  o::Real stopPower = 0.5*log(1.0 + 1.2288*reducedEnergy)/(reducedEnergy +
          0.1728*sqrt(reducedEnergy) + 0.008*pow(reducedEnergy, 0.1504));
  return stopPower;
}

//not used ? 
inline o::Real surfaceErosion(SCS* scs, o::Write<o::Real>& erosionData) {
  int atomZ = PARTICLE_Z;
  auto amu = PTCL_AMU;
  auto elCharge = ELECTRON_CHARGE; //1.60217662e-19
  auto protonMass = PROTON_MASS; //1.6737236e-27
  o::Real q = 18.6006;
  o::Real lambda = 2.2697;
  o::Real mu = 3.1273;
  o::Real Eth = 24.9885;
  o::Real targetZ = 74.0;
  o::Real targetM = 183.84;

  auto pid_scs = scs->get<PTC_ID>();
  auto vel_scs = scs->get<PTCL_VEL>();
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    auto ptcl = pid_scs(pid);
    auto screenLength = screeningLength(atomz, targetZ);
    auto vel = p::makeVector3(pid, vel_scs);
    o::Real stopPower = stoppingPower(vel, targetM, targetZ, screenLength);
    o::Real E0 = 0.5*amu*protonMass* 1/elCharge * p::osh_dot(vel, vel);
    o::Real term = pow((E0/Eth - 1), mu);
    o::Real Y0 = q*stopPower*term/(lambda + term);
    erosionData[pid] = Y0;
  };
  scs->parallel_for(lamb);
}

//TODO split this function/organize data access
//Note, elem_ids are index by pid=indexes, not scs member. Don't rebuild after search_mesh 
inline void surfaceReflection(o::Mesh& mesh, SCS* scs, o::Write<o::LO>& elem_ids) {
  int debug = 1;
  o::Real pi = CONSTANT_PI;
  o::LO fluxEA = SURFACE_FLUX_EA;
  o::LO bdryAtomZ = BOUNDARY_ATOM_Z;

  const auto coords = mesh.coords();
  const auto face_verts = mesh.ask_verts_of(2);
  const auto f2r_ptr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto f2r_elem = mesh.ask_up(o::FACE, o::REGION).ab2b;
  const auto side_is_exposed = mark_exposed_sides(&mesh);
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto down_r2fs = mesh.ask_down(3, 2).ab2b;

  auto pid_scs = scs->get<PTCL_ID>();
  auto tgt_scs = scs->get<PTCL_NEXT_POS>();
  auto pos_prev_scs = scs->get<PTCL_POS>();
  auto vel_scs = scs->get<PTCL_VEL>();
  auto weight_scs = scs->get<PTCL_WEIGHT>();

  auto& xpoints = gp.collisionPoints;
  auto& xfaces = gp.collisionPointFaceIds;
  //TODO check if this works, otherwise read and write as shown below
  //get mesh tag for boundary data id,xpt,vel. deep_copy doesn't work
  //auto xtag_w = deep_copy(mesh->get_array<o::Real>(o::FACE, "gridE"));
  auto gridE =  deep_copy(mesh->get_array<o::Real>(o::FACE, "gridE"));//TODO store subgrid data ?
  auto gridA =  deep_copy(mesh->get_array<o::Real>(o::FACE, "gridA"));//TODO store subgrid data ?
  auto sumPtclStrike =  deep_copy(mesh->get_array<o::Real>(o::FACE, "sumParticlesStrike"));
  auto sumWtStike =  deep_copy(mesh->get_array<o::Real>(o::FACE, "sumWeightStrike"));
  auto grossDep =  deep_copy(mesh->get_array<o::Real>(o::FACE, "grossDeposition"));
  auto grossEros_=  deep_copy(mesh->get_array<o::Real>(o::FACE, "grossErosion"));
  auto aveSputtYld =  deep_copy(mesh->get_array<o::Real>(o::FACE, "aveSputtYld"));  
  auto sputtYldCount =  deep_copy(mesh->get_array<o::Real>(o::FACE, "sputtYldCount")); 
  auto enerDist =  deep_copy(mesh->get_array<o::Real>(o::FACE, "energyDistribution")); //nEdist*nAdist
  auto sputtDist =  deep_copy(mesh->get_array<o::Real>(o::FACE, "sputtDistribution")); // nEdist*nAdist
  auto reflDist =  deep_copy(mesh->get_array<o::Real>(o::FACE, "reflDistribution")); //nEdist x nAdist

 /*
  auto xtag_r = mesh.get_array<o::Real>(o::FACE, "gridE"); 
  o::Write<o::Real> xtag_d(xtag_r.size());
  o::parallel_for(mesh.nfaces(), OMEGA_H_LAMBDA(int i) {
    xtag_d[i] = xtag_r[i];
  };);  //no tag for only bdry faces
 */
  auto lamb = SCS_LAMBDA(const int& e, const int& pid, const int& mask) {
    //mask is set for origin element, not for detected/exiting element
    if(mask >0 && elem_ids[pid]==-1) {
      auto elemId = e;
      auto fid = xfaces[ptcl];
      if(fid >= 0) {
        OMEGA_H_CHECK(side_is_exposed[fid]);
        auto ptcl = pid_scs(pid);
        auto pelem = p::elem_id_of_bdry_face_of_tet(fid, f2r_ptr, f2r_elem);
        if(elemId != pelem)
          elemId = pelem;
// need fid to fIndex map to access by bdryFaces[fIndex]:  
//auto fInd = get_bdry_face_index(o::LO fid, coords, side_is_exposed, );
        auto bdryAtomZ = xtag_d[fInd*dof+ATOMZ];
        auto vel = p::makeVector3(pid, vel_scs );
        o::Vector<3> xpt;
        for(o::LO i=0; i<3; ++i)
          xpt[i] = xpoints[ptcl*3+i];
        auto pos = p::makeVector3(pid, tgt_scs);
        o::Real dEdist = 0;
        o::Real dAdist = 0;
        o::LO AdistInd=0;
        o::LO EdistInd=0;
        if(fluxEA > 0 {
          dEdist = (Edist - E0dist)/nEdist;
          dAdist = (Adist - A0dist)/nAdist;
          AdistInd=0;
          EdistInd=0;
        }
        //firstColl(pid) = 1; //scs
        o::Real magPath = o::norm(vel);
        o::Real amu = ptcl_amu(pid); //scs
        o::Real E0 = 0.5*amu*1.6737236e-27*(magPath*magPath)/1.60217662e-19;
        if(E0 > 1000.0) 
          E0 = 990.0;
  
        auto elmId = p::elem_id_of_bdry_face_of_tet(fid, f2r_ptr, f2r_elem);
        auto surfNorm = p::face_normal_of_tet(fid, elmId, coords, mesh2verts, 
          face_verts, down_r2fs); //inwards ?
        o::Real magSurfNorm = o::norm(surfNorm);
        auto normVel = o::normalize(vel);
        o::Real ptclProj = p::osh_dot(normVel, surfNorm);
        o::Real thetaImpact = acos(ptclProj);
        if(thetaImpact > o::PI*0.5)
           thetaImpact = abs(thetaImpact - o::PI);
        thetaImpact = thetaImpact*180.0/o::PI;
        if(thetaImpact < 0) thetaImpact = 0;
    
        if(o::are_close(E0, 0))
          thetaImpact = 0;
        o::Real Y0;
        o::Real R0;
        if(bdryAtomZ > 0) {//TODO
          Y0 = p::interpolate2dField(readInData_d, rMin, zMin, dr, dz, nR, nZ, pos, true);

          interp2d(thetaImpact,log10(E0),nA_sputtRefCoeff, nE_sputtRefCoeff,
               A_sputtRefCoeff, Elog_sputtRefCoeff,spyl_surfaceModel);
          R0 = interp2d(thetaImpact,log10(E0),nA_sputtRefCoeff, nE_sputtRefCoeff,
               A_sputtRefCoeff, Elog_sputtRefCoeff,rfyl_surfaceModel);
        }
        else
        {
            Y0 = 0.0;
            R0 = 0.0;
        }

        o::Real totalYR=Y0+R0;
        //particle either reflects or deposits
        o::Real sputtProb = Y0/totalYR;
        int didReflect = 0;
        
        if(totalYR > 0) {
          if(r7 > sputtProb) { //reflect
            //resetting hitface
            xfaces[ptcl] = -1;
            // for next push
            elem_ids[pid] = elemId;

            didReflect = 1;
            aInterpVal = interp3d (r8,thetaImpact,log10(E0),
                      nA_sputtRefDistOut,nA_sputtRefDistIn,nE_sputtRefDistIn,
                                angleDistGrid01,A_sputtRefDistIn,
                                E_sputtRefDistIn,ADist_CDF_R_regrid);
            eInterpVal = interp3d ( r9,thetaImpact,log10(E0),
                       nE_sputtRefDistOutRef,nA_sputtRefDistIn,nE_sputtRefDistIn,
                                     energyDistGrid01Ref,A_sputtRefDistIn,
                                     E_sputtRefDistIn,EDist_CDF_R_regrid );
            //newWeight=(R0/(1.0f-sputtProb))*weight;
            newWeight = weight*(totalYR);
            if(FLUX_EA > 0) {
              EdistInd = floor((eInterpVal-E0dist)/dEdist);
              AdistInd = floor((aInterpVal-A0dist)/dAdist);
              if((EdistInd >= 0) && (EdistInd < nEdist) && 
                 (AdistInd >= 0) && (AdistInd < nAdist)) {
                reflDist[fid*nEdist*nAdist + EdistInd*nAdist + AdistInd] += newWeight; //TODO faceId serial ?
              }
            } 
            if(surface > 0) {
              auto dep = grossDep[fid];
              grossDep[fid] = dep + weight*(1.0-R0);
            }
          } else {//sputters                
            aInterpVal = interp3d(r8,thetaImpact,log10(E0),
                      nA_sputtRefDistOut,nA_sputtRefDistIn,nE_sputtRefDistIn,
                      angleDistGrid01,A_sputtRefDistIn, E_sputtRefDistIn,ADist_CDF_Y_regrid);
            eInterpVal = interp3d(r9,thetaImpact,log10(E0),
                    nE_sputtRefDistOut,nA_sputtRefDistIn,nE_sputtRefDistIn,
                    energyDistGrid01,A_sputtRefDistIn,E_sputtRefDistIn,EDist_CDF_Y_regrid);
                    newWeight = weight*totalYR;
            if(FLUX_EA > 0){
              EdistInd = floor((eInterpVal-E0dist)/dEdist);
              AdistInd = floor((aInterpVal-A0dist)/dAdist);
              if((EdistInd >= 0) && (EdistInd < nEdist) && 
                 (AdistInd >= 0) && (AdistInd < nAdist)) {
                sputtDist[fid*nEdist*nAdist + EdistInd*nAdist + AdistInd] += newWeight;
	            }
            }
            
            if(sputtProb == 0) 
              newWeight = 0;
            if(surface > 0) {
              grossDep[fid] += weight*(1.0-R0);
              grossEros[fid] += newWeight;
              aveSputtYld[fid] += Y0;
              if(weight > 0)
                sputtYldCount[fid] += 1;
            }
          }
          else {
            newWeight = 0;
            particles->hitWall[indx] = 2;
            if(surface > 0)
              grossDep[fid] += weight;
          }

          if(eInterpVal <= 0) {
            newWeight = 0;
            particles->hitWall[indx] = 2;
            if(surface > 0 && didReflect)
              grossDep[fid] += weight;            
          }
          if(surface) {
            sumWtStike[fid] += weight;
            sumPtclStrike[fid] += 1;

            if(FLUX_EA > 0) {
              EdistInd = floor((E0-E0dist)/dEdist);
              AdistInd = floor((thetaImpact-A0dist)/dAdist);
              if((EdistInd >= 0) && (EdistInd < nEdist) && 
                  (AdistInd >= 0) && (AdistInd < nAdist)) {
                enerDist[fid*nEdist*nAdist + EdistInd*nAdist + AdistInd] += weight;
              }
            }
          }

          if( bdryAtomZ > 0 && newWeight > 0) {
            ptcl_weight(pid) = newWeight;
            ptcl_hitWall(pid) = 0;
            ptcl_charge(pid) = 0;
            o::Real V0 = sqrt(2*eInterpVal*1.602e-19/(amu*1.66e-27));
            ptcl_newVelMag(pid) = V0;
            
            o::Vector<3> vSampled; 
            vSampled[0] = V0*sin(aInterpVal*3.1415/180)*cos(2.0*3.1415*r10);
            vSampled[1] = V0*sin(aInterpVal*3.1415/180)*sin(2.0*3.1415*r10);
            vSampled[2] = V0*cos(aInterpVal*3.1415/180);
            auto surfNorm = p::bdry_face_normal_of_tet(fid,coords,face_verts);
            auto surfPar = vtx1 - vtx2; // bdry face vtx
            auto Y = o::cross(surfNorm, surfPar);
            auto vSampled = vSampled[0] * surfPar + vSampled[1]*Y + vSampled[2]*surfNorm;

            auto newPos = -dbryInDir *surfNorm*vSampled; //boundaryVector[wallHit].inDir
            for(o::LO i=0; i<3; ++i)
              tgt_scs(pid,i) = newPos[i];

            auto prevPos = pos - 1e-6*dbryInDir *surfNorm;

            for(o::LO i=0; i<3; ++i)
              pos_prev_scs(pid,i) = prevPos[i];
          } else {
            particles->hitWall[indx] = 2.0;
          }
        }
        
      } //fid
    } //mask
  };
  scs->parallel_for(lamb);
  mesh.set_tag(o::FACE, "gridE", o::Reals(gridE)); //TODO store subgrid data ?
  mesh.set_tag(o::FACE, "gridA", o::Reals(gridA); //TODO store subgrid data ?
  mesh.set_tag(o::FACE, "sumParticlesStrike", o::Reals(sumPtclStrike);
  mesh.set_tag(o::FACE, "sumWeightStrike", o::Reals(sumWtStike);
  mesh.set_tag(o::FACE, "grossDeposition", o::Reals(grossDep);
  mesh.set_tag(o::FACE, "grossErosion", o::Reals(grossEros);
  mesh.set_tag(o::FACE, "aveSputtYld", o::Reals(aveSputtYld);
  mesh.set_tag(o::FACE, "sputtYldCount", o::Reals(sputtYldCount);
  mesh.set_tag(o::FACE, "energyDistribution", o::Reals(enerDist);
  mesh.set_tag(o::FACE, "sputtDistribution", o::Reals(sputtDist);
  mesh.set_tag(o::FACE, "reflDistribution", o::Reals(reflDist);
}





#endif
