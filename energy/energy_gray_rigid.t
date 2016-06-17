--Dimension parameters
local W,H = Dim("W",0), Dim("H",1)	--Image size
local N = 	Dim("N",2)				--Number of vertices
local S = 	Dim("S",3)				--One

--Problem parameters
local f_x = Param("f_x",float,0)
local f_y = Param("f_y",float,1)
local u_x = Param("u_x",float,2)
local u_y = Param("u_y",float,3)

local w_photometricSqrt = 	Param("w_photometricSqrt", float, 4)		--Photometric cost weight
local w_tempRigidTransSqrt = Param("w_tempRigidTransSqrt", float, 5)	--Temporal rigid translation regularisation weight

local RigidRot = 		Unknown("RigidRot", float3,{S},6)			--rigid rotation rotation.xyz <- unknown
local RigidTrans = 		Unknown("RigidTrans", float3,{S},7)			--rigid trnaslation trnaslation.xyz <- unknown

local MeshTrans = 			Array("MeshTrans", float3,{N},8)				--Per vertex displacement from template

local I_im = Array("I_im",float,{W,H},9) -- frame, sampled
local I_dx = Array("I_dx",float,{W,H},10) -- partials for frame
local I_dy = Array("I_dy",float,{W,H},11)

-- create a new math operator that samples from the image
local I = sampledimage(I_im, I_dx, I_dy)

local TemplateColors = 	Array("TemplateColors", float, {N},12)			--template shape: vertex.xyz
local TemplateShape = 	Array("TemplateShape", float3, {N},13)		--template shape: vertex.xyz
local PrevRigidTrans =	Array("PrevRigidTrans", float3,{S},14)		--previous rigid translation: translation.xyz
local Visibility = 		Array("Visibility", int, {N}, 15)				--Visibility
local G = Graph("G", 16, "v0", {N}, 17, "v1", {N}, 18)					--Graph

UsePreconditioner(true)

function Intensity(vertex)
    local x = vertex(0)
    local y = vertex(1)
    local z = vertex(2)
    local i = (f_x * x - u_x * z) / z
    local j = (f_y * y - u_y * z) / z
    return I(i,j)
end

function newVertex(v, dv, R, t)
	return Rotate3D(R, v + dv) + t
end

--Photometric Error Data term
local new_vertex = newVertex(TemplateShape(G.v0), MeshTrans(G.v0), RigidRot(0), RigidTrans(0))
local photometricCost = Intensity( new_vertex ) - TemplateColors(G.v0)
Energy( Select(Visibility(G.v0), w_photometricSqrt*photometricCost, 0) )

--Temporal Rigid Translation regularisation
local TempRigidTransCost = RigidTrans(0) - PrevRigidTrans(0)
Energy(w_tempRigidTransSqrt*TempRigidTransCost)
