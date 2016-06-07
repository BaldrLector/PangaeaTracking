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
local Im = 				Array("Im", double, {W,H}, 9) 					--Image Intensity
local TemplateColors = 	Image("TemplateColors", double, {N},10)			--template shape: vertex.xyz
local TemplateShape = 	Image("TemplateShape", opt.double3, {N},11)		--template shape: vertex.xyz
local PrevRigidTrans =	Image("PrevRigidTrans", opt.double3,{1},12)		--previous rigid translation: translation.xyz
local Visibility = 		Image("Visibility", int, {N}, 13)				--Visibility
local G = Graph("G", 14, "v0", {N}, 15, "v1", {N}, 16)					--Graph

UsePreconditioner(true)

function Intensity(v)
    local x = v(0)
    local y = v(1)
    local z = v(2)
    local i = (f_x * x - u_x * z) / z
    local j = (f_y * y - u_y * z) / z
    return Im(i,j)
end

function newVertex(v, dv, R, t)
	return Rotate3D(R, v + dv) + t
end

--Photometric Error Data term
local photometricCost= Intensity( newVertex(TemplateShape(0,0), Offset(0,0), RigidRot, RigidTrans) ) - TemplateColors(0,0)
Energy( Select(eq(Visibility(0), 0), 0, w_photometricSqrt*photometricCost) )

--Temporal Rigid Translation regularisation
local TempRigidTransCost = RigidTrans - PrevRigidTrans
Energy(w_tempRigidTransSqrt*TempRigidTransCost)
