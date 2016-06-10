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

-- function LinearInitAxis(x, size)
-- 	local ix = math.floor(x)
-- 	local x1, x2, dx
-- 	if ix < 0 then
--     	x1 = 0
--     	x2 = 0
--     	dx = 1.0
--   	elseif ix > (size - 2) then
-- 		x1 = size - 1
-- 		x2 = size - 1
-- 		dx = 1.0
-- 	else 
-- 		x1 = ix
-- 		x2 = ix + 1
-- 		dx = x2 - x
-- 	end
--     return x1, x2, dx
-- end

-- function SampledImage(I_im, I_gradX, I_gradY)
-- 	local x1, y1, x2, y2
-- 	local dx, dy
-- 	local im11, im12, im21, im22, im_value
-- 	local gradx11, gradx12, gradx21, gradx22, gradx_value
-- 	local grady11, grady12, grady21, grady22, grady_value

-- 	x1, x2, dx = LinearInitAxis(x, W:index())
-- 	y1, y2, dy = LinearInitAxis(y, H:index())

-- 	im11 = I_im(y1, x1)
-- 	im12 = I_im(y1, x2)
-- 	im21 = I_im(y2, x1)
-- 	im22 = I_im(y2, x2)

-- 	im_value = dy  * ( dx * im11 + (1.0 - dx) * im12 ) 
-- 		+ (1 - dy) * ( dx * im21 + (1.0 - dx) * im22 )

-- 	gradx11 = I_gradX(y1, x1)
-- 	gradx12 = I_gradX(y1, x2)
-- 	gradx21 = I_gradX(y2, x1)
-- 	gradx22 = I_gradX(y2, x2)

-- 	gradx_value = dy  * ( dx * gradx11 + (1.0 - dx) * gradx12 ) 
-- 		+ (1 - dy) * ( dx * gradx21 + (1.0 - dx) * gradx22 )

-- 	grady11 = I_gradY(y1, x1)
-- 	grady12 = I_gradY(y1, x2)
-- 	grady21 = I_gradY(y2, x1)
-- 	grady22 = I_gradY(y2, x2)

-- 	grady_value = dy  * ( dx * grady11 + (1.0 - dx) * grady12 ) 
-- 		+ (1 - dy) * ( dx * grady21 + (1.0 - dx) * grady22 )

-- 	return im_value, gradx_value, grady_value
-- end

-- create a new math operator that samples from the image
local I = SampledImage(I_im, Im_dx, Im_dy)

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
