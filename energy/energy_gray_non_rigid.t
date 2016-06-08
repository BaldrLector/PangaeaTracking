--Dimension parameters
local W,H = Dim("W",0), Dim("H",1)	--Image size
local N = 	Dim("N",2)				--Number of vertices

--Problem parameters
local f_x = Param("f_x",double,0)
local f_y = Param("f_y",double,1)
local u_x = Param("u_x",double,2)
local u_y = Param("u_y",double,3)

local w_photometricSqrt = 	Param("w_photometricSqrt", double, 4)	--Photometric cost weight
local w_tvSqrt = 			Param("w_tvSqrt", double, 5)				--TV regularisation weight
local w_arapSqrt = 			Param("w_arapSqrt", double, 6)			--ARAP regularisation weight
local w_tempShapeSqrt = 	Param("w_tempShapeSqrt", double, 7)		--Temporal shape regularisation weight

local Offset = 			Unknown("Offset", opt.double3,{N},8)			--vertex.xyz <- unknown
local Angle = 			Unknown("Angle", opt.double3,{N},9)			--vertex.xyz, rotation.xyz <- unknown

local RigidRot = 		Image("RigidRot", opt.double3,{1},10)		--rigid rotation rotation.xyz <- unknown
local RigidTrans = 		Image("RigidTrans", opt.double3,{1},11)	--rigid trnaslation trnaslation.xyz <- unknown

local I_im = Array("I_im",double,{W,H},12) -- frame, sampled
local I_dx = Array("I_dx",double,{W,H},13) -- partials for frame
local I_dy = Array("I_dy",double,{W,H},14)

 -- create a new math operator that samples from the image
local I = SampledImage(I_im, Im_dx, Im_dy)

local TemplateColors = 	Image("TemplateColors", double, {N},15)	--template shape: vertex.xyz
local TemplateShape = 	Image("TemplateShape", opt.double3, {N},16)	--template shape: vertex.xyz
local PrevOffset =		Image("PrevOffset", opt.double3,{N},17)		--previous vertices offset: vertex.xyz
local Visibility = 		Image("Visibility", uchar, {N}, 18)			--Visibility
local G = Graph("G", 19, "v0", {N}, 20, "v1", {N}, 21)				--Graph

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
local photometricCost = Intensity( newVertex(TemplateShape(0,0), Offset(0,0), RigidRot, RigidTrans) ) - TemplateColors(0,0)
Energy( Select(Visibility(0), w_photometricSqrt*photometricCost, 0) )

--TV regularization
local TVCost = Offset(G.v0) - Offset(G.v1)
Energy(w_tvSqrt*TVCost)

--ARAP regularization
local template_diff = TemplateShape(G.v0) - TemplateShape(G.v1)
local ARAPCost = template_diff 
               - Rotate3D( Angle(G.v0), template_diff + (Offset(G.v0) - Offset(G.v1)) )
Energy(w_arapSqrt*ARAPCost)

--Temporal Shape regularisation
local TempShapeCost = Offset(0) - PrevOffset(0)
Energy(w_tempShapeSqrt*TempShapeCost)
