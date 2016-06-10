--Dimension parameters
local W,H = Dim("W",0), Dim("H",1)	--Image size
local N = 	Dim("N",2)				--Number of vertices
local S = 	Dim("S",3)				--One

--Problem parameters
local f_x = Param("f_x",float,0)
local f_y = Param("f_y",float,1)
local u_x = Param("u_x",float,2)
local u_y = Param("u_y",float,3)

local w_photometricSqrt = 	Param("w_photometricSqrt", float, 4)	--Photometric cost weight
local w_tvSqrt = 			Param("w_tvSqrt", float, 5)				--TV regularisation weight
local w_arapSqrt = 			Param("w_arapSqrt", float, 6)			--ARAP regularisation weight
local w_tempShapeSqrt = 	Param("w_tempShapeSqrt", float, 7)		--Temporal shape regularisation weight

local MeshTrans = 			Unknown("MeshTrans", float3,{N},8)			--vertex.xyz <- unknown
local Angle = 			Unknown("Angle", float3,{N},9)			--vertex.xyz, rotation.xyz <- unknown

local RigidRot = 		Array("RigidRot", float3,{S},10)		--rigid rotation rotation.xyz <- unknown
local RigidTrans = 		Array("RigidTrans", float3,{S},11)	--rigid trnaslation trnaslation.xyz <- unknown

local I_im = Array("I_im",float,{W,H},12) -- frame, sampled
local I_dx = Array("I_dx",float,{W,H},13) -- partials for frame
local I_dy = Array("I_dy",float,{W,H},14)

local TemplateColors = 	Array("TemplateColors", float, {N},15)	--template shape: vertex.xyz
local TemplateShape = 	Array("TemplateShape", float3, {N},16)	--template shape: vertex.xyz
local PrevMeshTrans =	Array("PrevMeshTrans", float3,{N},17)		--previous vertices MeshTrans: vertex.xyz
local Visibility = 		Array("Visibility", int, {N}, 18)			--Visibility
local G = Graph("G", 19, "v0", {N}, 20, "v1", {N}, 21)				--Graph

UsePreconditioner(true)

-- create a new math operator that samples from the image
local I = SampledImage(I_im, Im_dx, Im_dy)

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
local new_vertex = newVertex(TemplateShape(G.v0), MeshTrans(G.v0), RigidRot(0), RigidTrans(0))
local photometricCost = Intensity(new_vertex) - TemplateColors(G.v0)
Energy( Select(Visibility(G.v0), w_photometricSqrt*photometricCost, 0) )

--TV regularization
local TVCost = MeshTrans(G.v0) - MeshTrans(G.v1)
Energy(w_tvSqrt*TVCost)

--ARAP regularization
local template_diff = TemplateShape(G.v0) - TemplateShape(G.v1)
local trans_diff = MeshTrans(G.v0) - MeshTrans(G.v1)
local ARAPCost = template_diff 
               - Rotate3D( Angle(G.v0), template_diff + trans_diff )
Energy(w_arapSqrt*ARAPCost)

--Temporal Shape regularisation
local TempShapeCost = MeshTrans(G.v0) - PrevMeshTrans(G.v0)
Energy(w_tempShapeSqrt*TempShapeCost)
