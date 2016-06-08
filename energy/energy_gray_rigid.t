--Dimension parameters
local W,H = Dim("W",0), Dim("H",1)	--Image size
local N = 	Dim("N",2)				--Number of vertices

--Problem parameters
local f_x = Param("f_x",double,0)
local f_y = Param("f_y",double,1)
local u_x = Param("u_x",double,2)
local u_y = Param("u_y",double,3)

local w_photometricSqrt = 	Param("w_photometricSqrt", double, 4)		--Photometric cost weight
local w_tempRigidTransSqrt = Param("w_tempRigidTransSqrt", double, 5)	--Temporal rigid translation regularisation weight

local RigidRot = 		Unknown("RigidRot", opt.double3,{1},6)			--rigid rotation rotation.xyz <- unknown
local RigidTrans = 		Unknown("RigidTrans", opt.double3,{1},7)			--rigid trnaslation trnaslation.xyz <- unknown

local Offset = 			Image("Offset", opt.double3,{N},8)				--vertex.xyz

local I_im = Array("I_im",double,{W,H},9) -- frame, sampled
local I_dx = Array("I_dx",double,{W,H},10) -- partials for frame
local I_dy = Array("I_dy",double,{W,H},11)

 -- create a new math operator that samples from the image
local I = SampledImage(I_im, Im_dx, Im_dy)

local TemplateColors = 	Image("TemplateColors", double, {N},12)			--template shape: vertex.xyz
local TemplateShape = 	Image("TemplateShape", opt.double3, {N},13)		--template shape: vertex.xyz
local PrevRigidTrans =	Image("PrevRigidTrans", opt.double3,{1},14)		--previous rigid translation: translation.xyz
local Visibility = 		Image("Visibility", uchar, {N}, 15)				--Visibility

UsePreconditioner(true)

function Intensity(v)
    local x = v(0)
    local y = v(1)
    local z = v(2)
    local i = (f_x * x - u_x * z) / z
    local j = (f_y * y - u_y * z) / z
    return I(i,j)
end

function newVertex(v, dv, R, t)
	return Rotate3D(R, v + dv) + t
end

--Photometric Error Data term
local photometricCost= Intensity( newVertex(TemplateShape(0,0), Offset(0,0), RigidRot, RigidTrans) ) - TemplateColors(0,0)
Energy( Select(Visibility(0), w_photometricSqrt*photometricCost, 0) )

--Temporal Rigid Translation regularisation
local TempRigidTransCost = RigidTrans - PrevRigidTrans
Energy(w_tempRigidTransSqrt*TempRigidTransCost)
